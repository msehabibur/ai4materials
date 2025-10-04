# tabs/elastic_tab.py
from __future__ import annotations

import json
from typing import Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import torch

from jobflow import run_locally, SETTINGS
from monty.json import jsanitize
from pymatgen.core import Structure
from pymatgen.analysis.elasticity.elastic import ElasticTensor

EV_PER_ANG3_TO_GPA = 160.21766208

# ------------------------ Progress ------------------------
class _StepProgress:
    def __init__(self, total_steps: int, label: str = "Working…"):
        self.total = max(int(total_steps), 1)
        self.curr = 0
        self._pct = st.empty()
        self._status = st.empty()
        self._bar = st.progress(0, text=label)

    def tick(self, msg: str):
        self.curr = min(self.curr + 1, self.total)
        pct = int(self.curr / self.total * 100)
        self._status.write(f"**{msg}**")
        self._bar.progress(pct)
        self._pct.caption(f"{self.curr}/{self.total} steps")

    def finish(self, ok: bool = True):
        if ok:
            self._bar.progress(100, text="Done")

# ------------------------ Utils ------------------------
def _to_plain(x):
    try:
        return jsanitize(x, strict=False)
    except Exception:
        return x

def _is_6x6_numeric(x) -> bool:
    try:
        a = np.array(x, dtype=float)
        return a.shape == (6, 6) and np.isfinite(a).all()
    except Exception:
        return False

def _walk_paths(tree: Any, path: str = "$"):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _walk_paths(v, f"{path}.{k}")
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            yield from _walk_paths(v, f"{path}[{i}]")
    else:
        yield (path, tree)

def _maybe_convert_to_gpa(C: np.ndarray) -> Tuple[np.ndarray, str]:
    C = np.array(C, dtype=float).reshape(6, 6)
    # Typical GPa magnitudes ~10–300; if much smaller, assume eV/Å^3
    if np.max(np.abs(C)) < 20.0:
        return C * EV_PER_ANG3_TO_GPA, "eV/Å³→GPa"
    return C, "GPa"

def _extract_tensor_from_output_dict(out: dict) -> Optional[np.ndarray]:
    props = out.get("physical_properties") or {}
    candidates: List[Any] = [
        props.get("elastic_tensor_voigt"),
        props.get("C_ij"),
        out.get("elastic_tensor_voigt"),
        out.get("C_ij"),
        out.get("elastic_tensor"),
    ]
    for C in candidates:
        if C is not None and _is_6x6_numeric(C):
            return np.array(C, dtype=float).reshape(6, 6)
    return None

def _query_elastic_from_store() -> Tuple[Optional[np.ndarray], Optional[str], Optional[dict]]:
    store = SETTINGS.JOB_STORE
    if store is None:
        return None, None, None
    try:
        store.connect()
    except Exception:
        return None, None, None

    names = ("fit_elastic_tensor", "elastic_fit", "get_elastic_tensor")
    last_doc = None
    for name in names:
        try:
            doc = store.query_one(
                {"name": name},
                properties=["output", "completed_at", "name", "uuid"],
                sort={"completed_at": -1},
                load=True,
            )
        except Exception:
            doc = None
        if isinstance(doc, dict):
            last_doc = doc
            out = doc.get("output") or {}
            # try the common locations
            for key in (
                "physical_properties.elastic_tensor_voigt",
                "physical_properties.C_ij",
                "elastic_tensor_voigt",
                "C_ij",
                "elastic_tensor",
            ):
                node = out
                ok = True
                for p in key.split("."):
                    if isinstance(node, dict) and p in node:
                        node = node[p]
                    else:
                        ok = False
                        break
                if ok and _is_6x6_numeric(node):
                    return np.array(node, float).reshape(6, 6), f"JobStore:{name}.output.{key}", doc
    return None, None, last_doc

def _deep_find_elastic(tree: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
    # Quick checks
    if isinstance(tree, dict):
        for key in ("output", "result", "results", "data", "physical_properties"):
            node = tree.get(key)
            if isinstance(node, dict):
                for k in ("elastic_tensor_voigt", "C_ij", "elastic_tensor"):
                    v = node.get(k)
                    if _is_6x6_numeric(v):
                        return np.array(v, float).reshape(6, 6), f"rs.{key}.{k}"

    # Deep search
    hits: List[Tuple[str, np.ndarray]] = []
    for p, v in _walk_paths(tree):
        if _is_6x6_numeric(v):
            hits.append((p, np.array(v, float).reshape(6, 6)))
    if not hits:
        return None, None
    # Prefer paths that mention 'elastic'
    hits.sort(key=lambda t: ("elastic" not in t[0], len(t[0])))
    return hits[0][1], hits[0][0]

def _render_elastic(C_raw: np.ndarray, where: Optional[str], show_where: bool):
    C_gpa, conv = _maybe_convert_to_gpa(C_raw)
    et = ElasticTensor.from_voigt(C_gpa)

    if show_where and where:
        st.caption(f"Found elastic tensor at: `{where}` (conversion: {conv})")

    st.subheader("Elastic constants (Voigt 6×6, GPa)")
    df = pd.DataFrame(
        C_gpa,
        index=[f"C{i+1}*" for i in range(6)],
        columns=[f"*{j+1}" for j in range(6)],
    )
    st.dataframe(df.style.format("{:.2f}"), use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bulk (K_VRH)", f"{et.k_vrh:.2f} GPa")
    c2.metric("Shear (G_VRH)", f"{et.g_vrh:.2f} GPa")
    c3.metric("Young’s (E)", f"{et.y_modulus:.2f} GPa")
    c4.metric("Poisson (ν)", f"{et.poisson_ratio:.3f}")
    c5.metric("Anisotropy (Aᵁ)", f"{et.universal_anisotropy:.2f}")

    payload = {
        "units": "GPa",
        "unit_conversion": conv,
        "C_voigt": C_gpa.tolist(),
        "K_VRH": float(et.k_vrh),
        "G_VRH": float(et.g_vrh),
        "E_VRH": float(et.y_modulus),
        "nu": float(et.poisson_ratio),
        "A_U": float(et.universal_anisotropy),
        "source_path": where or "",
    }
    st.download_button(
        "⬇️ Elastic constants (JSON)",
        json.dumps(payload, indent=2).encode(),
        "elastic_constants.json",
    )

# ------------------------ Main tab ------------------------
def elastic_tab(pmg_obj: Structure | None):
    st.subheader("🪨 Elastic Constants — ML Force Field (read & default-run)")

    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer to compute elastic constants.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        precision = st.radio("Precision", ["float64 (recommended)", "float32 (fast)"], index=0, key="el_prec")
    with col2:
        show_where = st.toggle("Debug (show key path)", value=False)
    with col3:
        load_latest = st.button("Load Latest From JobStore", use_container_width=True)

    # Precision (avoid MACE down-cast for relax/elastic)
    torch.set_default_dtype(torch.float64 if precision.startswith("float64") else torch.float32)

    # Show cached results across reruns
    if "elastic_C_cached" in st.session_state:
        _render_elastic(st.session_state["elastic_C_cached"], st.session_state.get("elastic_where"), show_where)

    # ----- Reader-only path -----
    if load_latest:
        C, where, _doc = _query_elastic_from_store()
        if C is None:
            st.error("No elastic tensor found in JobStore. Make sure the elastic workflow wrote results to the store.")
            return
        st.session_state["elastic_C_cached"] = C
        st.session_state["elastic_where"] = where
        _render_elastic(C, where, show_where)
        return

    st.divider()
    st.caption("Or run with default settings (same as before):")

    # ----- Default-run path (no unsupported kwargs) -----
    run_btn = st.button("Run Elastic Workflow (default)", type="primary")
    if not run_btn:
        return

    steps = _StepProgress(6, "Elastic workflow running…")
    try:
        steps.tick("Building workflow")
        try:
            # IMPORTANT: no unsupported kwargs here
            from atomate2.forcefields.flows.elastic import ElasticMaker
        except Exception:
            st.error("ElasticMaker not found (atomate2.forcefields.flows.elastic). Please update atomate2.")
            return

        maker = ElasticMaker()              # <-- KEEP LIKE BEFORE (no max_strain / n_steps)
        flow = maker.make(structure=pmg_obj)

        steps.tick("Executing locally")
        rs = run_locally(flow, create_folders=True, ensure_success=False)

        steps.tick("Querying JobStore")
        C, where, _doc = _query_elastic_from_store()

        if C is None:
            steps.tick("Scanning in-memory results")
            C, where = _deep_find_elastic(_to_plain(rs))

        if C is None:
            steps.finish(False)
            st.error("Elastic tensor not found (no 'elastic_tensor_voigt' / 'C_ij' in outputs).")
            return

        st.session_state["elastic_C_cached"] = C
        st.session_state["elastic_where"] = where

        steps.tick("Rendering results")
        _render_elastic(C, where, show_where)

        steps.finish(True)
        st.success("Elastic constants computed ✅")

    except Exception as e:
        steps.finish(False)
        st.error(f"Elastic workflow failed: {e}")
        with st.expander("Details"):
            st.exception(e)

__all__ = ["elastic_tab"]
