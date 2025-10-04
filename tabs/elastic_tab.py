# tabs/elastic_tab.py
from __future__ import annotations

import json
from typing import Any, Optional, Tuple, List, Union

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

def _maybe_convert_to_gpa(C: np.ndarray) -> Tuple[np.ndarray, str]:
    C = np.array(C, dtype=float).reshape(6, 6)
    # Typical GPa magnitudes are ~10–300; if much smaller, assume eV/Å^3
    if np.max(np.abs(C)) < 20.0:
        return C * EV_PER_ANG3_TO_GPA, "eV/Å³→GPa"
    return C, "GPa"

def _cij_dict_to_matrix(d: dict) -> Optional[np.ndarray]:
    """
    Accepts dicts like {'C11':..., 'C12':..., ..., 'C66':...} (case-insensitive),
    fills symmetric 6x6 Voigt tensor.
    """
    if not isinstance(d, dict):
        return None
    D = {k.lower(): v for k, v in d.items()}
    needed = [f"c{i}{j}" for i in range(1,7) for j in range(1,7) if i<=j]  # upper triangle keys
    if not any(k in D for k in needed):
        return None
    C = np.zeros((6,6), float)
    for i in range(1,7):
        for j in range(i,7):
            key = f"c{i}{j}"
            if key in D:
                C[i-1, j-1] = float(D[key])
                C[j-1, i-1] = float(D[key])
    # sanity: nonzero diagonal?
    if not np.any(np.diag(C)):
        return None
    return C

def _triplet_list_to_matrix(lst: List[dict]) -> Optional[np.ndarray]:
    """
    Accepts list like [{'i':1,'j':2,'value':..}, ...] or [i,j,val] triplets.
    """
    try:
        C = np.zeros((6,6), float)
        filled = False
        for item in lst:
            if isinstance(item, dict):
                i, j, v = int(item["i"]), int(item["j"]), float(item["value"])
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                i, j, v = int(item[0]), int(item[1]), float(item[2])
            else:
                continue
            if 1 <= i <= 6 and 1 <= j <= 6:
                C[i-1, j-1] = v
                C[j-1, i-1] = v
                filled = True
        if filled:
            return C
    except Exception:
        pass
    return None

def _walk_paths(tree: Any, path: str = "$"):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _walk_paths(v, f"{path}.{k}")
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            yield from _walk_paths(v, f"{path}[{i}]")
    else:
        yield (path, tree)

def _extract_tensor_from_output_dict(out: dict) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Try multiple common locations/representations inside a job 'output' dict.
    Returns (C, path).
    """
    # 1) Direct voigt array
    props = out.get("physical_properties") or {}
    candidates: List[Tuple[str, Any]] = [
        ("output.physical_properties.elastic_tensor_voigt", props.get("elastic_tensor_voigt")),
        ("output.elastic_tensor_voigt", out.get("elastic_tensor_voigt")),
    ]
    # 2) nested 'elastic_tensor': {'voigt': [[...]]}
    et_props = props.get("elastic_tensor") if isinstance(props, dict) else None
    et_out = out.get("elastic_tensor") if isinstance(out, dict) else None
    candidates += [
        ("output.physical_properties.elastic_tensor.voigt", isinstance(et_props, dict) and et_props.get("voigt")),
        ("output.elastic_tensor.voigt", isinstance(et_out, dict) and et_out.get("voigt")),
    ]
    # 3) C_ij synonyms
    candidates += [
        ("output.physical_properties.C_ij", props.get("C_ij")),
        ("output.C_ij", out.get("C_ij")),
    ]
    # 4) dict of C11..C66
    for label, obj in list(candidates):
        if isinstance(obj, dict):
            m = _cij_dict_to_matrix(obj)
            if m is not None:
                return m, label
    # 5) list of triplets
    for label, obj in list(candidates):
        if isinstance(obj, list):
            m = _triplet_list_to_matrix(obj)
            if m is not None:
                return m, label
    # 6) plain 6x6 anywhere among candidates
    for label, obj in candidates:
        if _is_6x6_numeric(obj):
            return np.array(obj, float).reshape(6, 6), label
    return None, None

def _deep_find_elastic(tree: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Deep scan a nested dict/list for any recognizable elastic tensor form.
    Order:
      - known containers/keys
      - 6x6 numeric array anywhere
      - dict C11..C66
      - list of triplets
    """
    # First: check some likely containers quickly
    if isinstance(tree, dict):
        for key in ("output", "result", "results", "data", "physical_properties"):
            node = tree.get(key)
            if isinstance(node, dict):
                C, where = _extract_tensor_from_output_dict(node if key == "output" else {"physical_properties": node})
                if C is not None:
                    return C, f"rs.{key}.{where.split('.',1)[-1]}"
                # direct keys inside container
                for k in ("elastic_tensor_voigt", "C_ij", "elastic_tensor"):
                    v = node.get(k)
                    if v is not None:
                        if _is_6x6_numeric(v):
                            return np.array(v, float).reshape(6, 6), f"rs.{key}.{k}"
                        if isinstance(v, dict):
                            m = _cij_dict_to_matrix(v) or _extract_tensor_from_output_dict({"elastic_tensor": v})[0]
                            if m is not None:
                                return m, f"rs.{key}.{k}"
                        if isinstance(v, list):
                            m = _triplet_list_to_matrix(v)
                            if m is not None:
                                return m, f"rs.{key}.{k}"

    # Full deep walk: prefer paths that mention 'elastic'
    array_hits: List[Tuple[str, np.ndarray]] = []
    dict_hits: List[Tuple[str, np.ndarray]] = []
    triplet_hits: List[Tuple[str, np.ndarray]] = []
    for p, v in _walk_paths(tree):
        if _is_6x6_numeric(v):
            array_hits.append((p, np.array(v, float).reshape(6, 6)))
        elif isinstance(v, dict):
            m = _cij_dict_to_matrix(v)
            if m is not None:
                dict_hits.append((p, m))
            else:
                # nested 'voigt'
                if "voigt" in v and _is_6x6_numeric(v.get("voigt")):
                    dict_hits.append((p+".voigt", np.array(v["voigt"], float).reshape(6,6)))
        elif isinstance(v, list):
            m = _triplet_list_to_matrix(v)
            if m is not None:
                triplet_hits.append((p, m))

    def _best(hits: List[Tuple[str, np.ndarray]]) -> Optional[Tuple[str, np.ndarray]]:
        if not hits:
            return None
        hits.sort(key=lambda t: ("elastic" not in t[0].lower(), len(t[0])))
        return hits[0]

    for bag in (dict_hits, triplet_hits, array_hits):
        best = _best(bag)
        if best:
            return best[1], best[0]
    return None, None

def _query_elastic_from_store() -> Tuple[Optional[np.ndarray], Optional[str], Optional[dict]]:
    """
    Pull latest doc from JobStore and deep-scan it entirely (not just output),
    since schemas differ across versions.
    """
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
            # load the full doc to allow deep scan
            doc = store.query_one({"name": name}, properties=["*"], sort={"completed_at": -1}, load=True)
        except Exception:
            doc = None
        if isinstance(doc, dict):
            last_doc = doc
            C, where = _deep_find_elastic(doc)
            if C is not None:
                return C, f"JobStore:{name}:{where or 'full_doc'}", doc
    return None, None, last_doc

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
        "C_voigt": C_gpa.tolist(),
        "unit_conversion": conv,
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

    # ----- Default-run path (KEEP LIKE BEFORE) -----
    run_btn = st.button("Run Elastic Workflow (default)", type="primary")
    if not run_btn:
        return

    steps = _StepProgress(6, "Elastic workflow running…")
    try:
        steps.tick("Building workflow")
        try:
            from atomate2.forcefields.flows.elastic import ElasticMaker  # no kwargs
        except Exception:
            st.error("ElasticMaker not found (atomate2.forcefields.flows.elastic). Please update atomate2.")
            return

        maker = ElasticMaker()  # KEEP LIKE BEFORE (no unsupported kwargs)
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
            st.error("Elastic tensor not found in outputs. Turn on Debug to inspect paths.")
            # Minimal peek to help diagnose (without spamming)
            if show_where:
                plain = _to_plain(rs)
                top_keys = list(plain.keys()) if isinstance(plain, dict) else [type(plain).__name__]
                st.caption("Top-level keys of in-memory result:")
                st.json(top_keys)
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
