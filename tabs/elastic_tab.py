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
    """Yield (path, value) for all leaves in dict/list trees."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _walk_paths(v, f"{path}.{k}")
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            yield from _walk_paths(v, f"{path}[{i}]")
    else:
        yield (path, tree)

def _maybe_convert_to_gpa(C: np.ndarray) -> Tuple[np.ndarray, str]:
    """Heuristic convert eV/Å^3 → GPa if values look too small for GPa."""
    C = np.array(C, dtype=float).reshape(6, 6)
    if np.max(np.abs(C)) < 20.0:  # typical C_ij in GPa are 10–300+
        return C * EV_PER_ANG3_TO_GPA, "eV/Å³→GPa"
    return C, "GPa"

def _extract_tensor_from_output_dict(out: dict) -> Optional[np.ndarray]:
    """Try common locations for the elastic tensor inside a job 'output' dict."""
    props = out.get("physical_properties") or {}
    candidates: List[Any] = [
        props.get("elastic_tensor_voigt"),
        props.get("C_ij"),
        out.get("elastic_tensor_voigt"),
        out.get("C_ij"),
        out.get("elastic_tensor"),  # some forks use this
    ]
    for C in candidates:
        if C is not None and _is_6x6_numeric(C):
            return np.array(C, dtype=float).reshape(6, 6)
    return None

def _query_elastic_from_store(debug: bool = False) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[str]]:
    """Fetch latest elastic tensor from JobStore; return (C, raw_doc_for_debug, where_path)."""
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
            # check common keys directly
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
                    return np.array(node, float).reshape(6, 6), (doc if debug else None), f"JobStore:{name}.output.{key}"
    return None, (last_doc if debug else None), None

def _deep_find_elastic(tree: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Scan nested dict/list for a 6x6 elastic tensor under any key; return (C, path)."""
    # quick checks of likely containers
    if isinstance(tree, dict):
        for key in ("output", "result", "results", "data", "physical_properties"):
            node = tree.get(key)
            if isinstance(node, dict):
                for k in ("elastic_tensor_voigt", "C_ij", "elastic_tensor"):
                    v = node.get(k)
                    if _is_6x6_numeric(v):
                        return np.array(v, float).reshape(6, 6), f"rs.{key}.{k}"

    # deep search every leaf
    hits: List[Tuple[str, np.ndarray]] = []
    for p, v in _walk_paths(tree):
        if _is_6x6_numeric(v):
            hits.append((p, np.array(v, float).reshape(6, 6)))
    if not hits:
        return None, None
    # prefer paths that mention 'elastic'
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
    st.subheader("🪨 Elastic Constants — ML Force Field")

    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer to compute elastic constants.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        precision = st.radio("Precision", ["float64 (recommended)", "float32 (fast)"], index=0, key="el_prec")
    with col2:
        max_strain = st.number_input(
            "Max strain",
            min_value=0.0025,
            max_value=0.04,
            value=0.01,
            step=0.0025,
            help="Peak engineering strain used for finite deformations.",
        )
    with col3:
        n_steps = st.selectbox("Deformation steps", [6, 10, 12], index=0)
    with col4:
        debug = st.toggle("Debug", value=False, help="Show where the tensor was found / key paths.")

    # Set dtype upfront to avoid MACE down-cast during relax/elastic
    torch.set_default_dtype(torch.float64 if precision.startswith("float64") else torch.float32)

    # Show cached tensor on reruns (Streamlit re-exec safe)
    if "elastic_C_cached" in st.session_state:
        _render_elastic(st.session_state["elastic_C_cached"], st.session_state.get("elastic_where"), show_where=debug)

    run_btn = st.button("Run Elastic Workflow", type="primary", key="el_run")

    if not run_btn:
        return

    steps = _StepProgress(8, "Elastic workflow running…")

    try:
        # 1) Build the flow
        steps.tick("Building workflow")
        try:
            from atomate2.forcefields.flows.elastic import ElasticMaker  # newer atomate2
        except Exception:
            st.error("ElasticMaker not found (atomate2.forcefields.flows.elastic). Please update atomate2.")
            return

        flow = ElasticMaker(max_strain=float(max_strain), n_steps=int(n_steps)).make(structure=pmg_obj)

        # 2) Execute
        steps.tick("Executing locally")
        rs = run_locally(flow, create_folders=True, ensure_success=False)

        # 3) Try JobStore first
        steps.tick("Querying JobStore")
        C, store_doc, where = _query_elastic_from_store(debug=debug)

        # 4) Fallback to in-memory result scan
        if C is None:
            steps.tick("Scanning in-memory results")
            C, where = _deep_find_elastic(_to_plain(rs))

        # 5) Render
        if C is None:
            steps.finish(False)
            st.error("Elastic tensor not found (no 'elastic_tensor_voigt' / 'C_ij' in outputs).")
            if debug:
                if store_doc is not None:
                    out = store_doc.get("output") or {}
                    st.caption("Last JobStore doc output keys:")
                    st.json(list(out.keys()))
                    st.caption("physical_properties keys:")
                    st.json(list((out.get("physical_properties") or {}).keys()))
                else:
                    st.caption("No JobStore doc available; your run may be in-memory only.")
            return

        st.session_state["elastic_C_cached"] = np.array(C, dtype=float).reshape(6, 6)
        st.session_state["elastic_where"] = where

        steps.tick("Rendering results")
        _render_elastic(st.session_state["elastic_C_cached"], where, show_where=debug)

        steps.finish(True)
        st.success("Elastic constants computed ✅")

        if debug and store_doc is not None:
            st.caption("Debug: Store doc summary")
            st.json({
                "store_job_name": store_doc.get("name"),
                "store_uuid": store_doc.get("uuid"),
            })

    except Exception as e:
        steps.finish(False)
        st.error(f"Elastic workflow failed: {e}")
        with st.expander("Details"):
            st.exception(e)

__all__ = ["elastic_tab"]
