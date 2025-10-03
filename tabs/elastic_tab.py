from __future__ import annotations
import io
import json
import csv
from typing import Any, Dict, Tuple, Optional, List, Union
import numbers
import numpy as np
import streamlit as st
from pymatgen.core import Structure

from monty.json import jsanitize
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import MACERelaxMaker
from jobflow import run_locally


# ---------- utils ----------
def _to_plain(obj: Any) -> Any:
    """Convert monty/MSON/typed objects to plain Python (dict/list/num/str)."""
    try:
        return jsanitize(obj, strict=False)
    except Exception:
        if hasattr(obj, "as_dict"):
            try:
                return jsanitize(obj.as_dict(), strict=False)
            except Exception:
                return obj
        return obj


def _to_gpa(val: Any):
    """Convert values that look like Pascals to GPa; recurse through lists/dicts."""
    if val is None:
        return None
    if isinstance(val, dict):
        return {k: _to_gpa(v) for k, v in val.items()}
    arr = np.array(val, dtype=float)
    # Heuristic: if magnitudes are large, assume Pa and convert to GPa.
    if np.nanmax(np.abs(arr)) > 1e5:
        arr = arr * 1e-9
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def _pack_csv(C: np.ndarray, props: dict) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Elastic tensor C (GPa) — 6×6 (Voigt/IEEE)"])
    for row in C:
        w.writerow([f"{v:.6f}" for v in row])
    w.writerow([])
    w.writerow(["Property", "Value"])
    for k in [
        "k_voigt", "k_reuss", "k_vrh",
        "g_voigt", "g_reuss", "g_vrh",
        "y_mod", "homogeneous_poisson", "universal_anisotropy",
    ]:
        v = props.get(k, None)
        w.writerow([k, f"{v:.6f}" if isinstance(v, (int, float)) else "—"])
    return buf.getvalue().encode()


# ---------- robust extraction ----------
def _looks_like_6x6(mat: Any) -> bool:
    """True if mat appears to be a 6×6 numeric array/list."""
    if isinstance(mat, (list, tuple)) and len(mat) == 6:
        try:
            rows = list(mat)
            if all(isinstance(r, (list, tuple)) and len(r) == 6 for r in rows):
                # check numeric
                for r in rows:
                    for x in r:
                        if not isinstance(x, numbers.Number):
                            return False
                return True
        except Exception:
            return False
    return False


def _maybe_tensor_anywhere(d: Any) -> Optional[List[List[float]]]:
    """
    Extremely permissive search:
    - Known keys: elastic_tensor.{ieee_format, voigt, matrix, C_ij, C}
    - Flat keys: C_ij, C, tensor, elastic, elastic_voigt
    - If not found by keys, find the FIRST 6×6 numeric matrix anywhere.
    """
    # Depth-first search
    stack: List[Any] = [d]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            # Known shapes
            et = cur.get("elastic_tensor")
            if isinstance(et, dict):
                for k in ("ieee_format", "voigt", "matrix", "C_ij", "C"):
                    if k in et and _looks_like_6x6(et[k]):
                        return _to_gpa(et[k])  # Pa->GPa if needed
            for k in ("C_ij", "C", "tensor", "elastic", "elastic_voigt"):
                if k in cur and _looks_like_6x6(cur[k]):
                    return _to_gpa(cur[k])
            # Push nested dict-like containers
            for child_key in ("output", "data", "result", "results", "elastic_data", "elastic"):
                if child_key in cur:
                    stack.append(cur[child_key])
            # Push all values for brute-force scan
            stack.extend(list(cur.values()))
        elif isinstance(cur, (list, tuple)):
            stack.extend(list(cur))
        else:
            # Primitive — ignore
            pass
    # Secondary pass: brute-force matrix hunt
    def walk_for_matrix(x: Any) -> Optional[List[List[float]]]:
        if _looks_like_6x6(x):
            return _to_gpa(x)
        if isinstance(x, dict):
            for v in x.values():
                got = walk_for_matrix(v)
                if got is not None:
                    return got
        elif isinstance(x, (list, tuple)):
            for v in x:
                got = walk_for_matrix(v)
                if got is not None:
                    return got
        return None
    return walk_for_matrix(d)


def _props_from_any(d: Any) -> dict:
    """Best-effort to extract derived properties dict from near-by keys."""
    if isinstance(d, dict):
        for k in ("derived_properties", "properties"):
            if k in d and isinstance(d[k], dict):
                return _to_gpa(d[k])
        # Search nested spots
        for k in ("output", "data", "result", "results", "elastic_data", "elastic"):
            if k in d and isinstance(d[k], dict):
                found = _props_from_any(d[k])
                if found:
                    return found
    return {}


def _extract_elastic(rs: Union[Dict[str, Any], List[Any]], flow_obj: Any) -> Tuple[np.ndarray, dict, dict, Any, list]:
    """
    Resolve flow outputs, sanitize, extract tensor (C 6×6) and props.
    Returns: (C ndarray, props dict, meta dict, resolved_plain, samples)
    """
    meta = {"found_in": None, "jobs_seen": []}

    # Pass 0: resolve flow.output against responses
    resolved_plain = None
    try:
        resolved = flow_obj.output.resolve(rs)
        resolved_plain = _to_plain(resolved)
        C = _maybe_tensor_anywhere(resolved_plain)
        if C is not None:
            P = _props_from_any(resolved_plain)
            meta["found_in"] = {"scope": "flow.output.resolve(rs)"}
            return np.array(C, dtype=float), P, meta, resolved_plain, []
    except Exception:
        pass  # continue

    # Normalize rs iterable
    items = list(rs.items()) if isinstance(rs, dict) else list(enumerate(rs))

    # Pass 1: per-job outputs (sanitize) — collect 5 samples for UI
    samples = []
    for i, (jid, rec) in enumerate(items):
        name = getattr(rec, "name", None)
        meta["jobs_seen"].append({"id": str(jid), "name": name})
        for field in ("output", "metadata"):
            obj = getattr(rec, field, None)
            if obj is None:
                continue
            plain = _to_plain(obj)
            if field == "output" and len(samples) < 5:
                samples.append({"job_id": str(jid), "name": name, "output_sample": plain})
            C = _maybe_tensor_anywhere(plain)
            if C is not None:
                P = _props_from_any(plain)
                meta["found_in"] = {"job_id": str(jid), "field": field}
                return np.array(C, dtype=float), P, meta, resolved_plain, samples

    # Pass 2: whole responses deep-scan
    whole_plain = _to_plain(rs)
    C = _maybe_tensor_anywhere(whole_plain)
    if C is not None:
        P = _props_from_any(whole_plain)
        meta["found_in"] = {"scope": "responses-deep-scan"}
        return np.array(C, dtype=float), P, meta, resolved_plain, samples

    raise RuntimeError("Elastic results not found in job outputs.")


# ---------- render ----------
def _render_payload(payload: dict):
    C = np.array(payload["C_GPa"], dtype=float)
    P = payload["props"]

    st.markdown("**Elastic tensor C (GPa, 6×6 Voigt/IEEE)**")
    st.dataframe([[f"{v:.6f}" for v in row] for row in C], use_container_width=True)

    cA, cB, cC = st.columns(3)
    cA.metric("K_VRH (GPa)", f"{P.get('k_vrh', float('nan')):.3f}" if P.get("k_vrh") is not None else "—")
    cB.metric("G_VRH (GPa)", f"{P.get('g_vrh', float('nan')):.3f}" if P.get("g_vrh") is not None else "—")
    cC.metric("E (GPa)",     f"{P.get('y_mod', float('nan')):.3f}" if P.get("y_mod") is not None else "—")

    cD, cE = st.columns(2)
    po = P.get("homogeneous_poisson", None)
    au = P.get("universal_anisotropy", None)
    cD.metric("Poisson ν", f"{po:.3f}" if isinstance(po, (int, float)) else "—")
    cE.metric("Anisotropy AU", f"{au:.3f}" if isinstance(au, (int, float)) else "—")

    json_bytes = json.dumps(payload, indent=2).encode()
    csv_bytes = _pack_csv(C, P)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ JSON", json_bytes, "elastic_results.json")
    d2.download_button("⬇️ CSV", csv_bytes, "elastic_results.csv")


# ---------- tab ----------
def elastic_tab(pmg_obj: Structure | None):
    st.subheader("🧱 Elastic — default flow (MACE)")
    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer first.")
        return

    # State
    if "elastic_running" not in st.session_state:
        st.session_state.elastic_running = False
    if "elastic_payload" not in st.session_state:
        st.session_state.elastic_payload = None
    if "elastic_debug" not in st.session_state:
        st.session_state.elastic_debug = None

    disabled = st.session_state.elastic_running

    c1, c2 = st.columns(2)
    with c1:
        fmax = st.number_input(
            "Relax fmax (eV/Å)",
            min_value=1e-8,
            max_value=1e-1,
            value=1e-5,
            step=1e-8,
            format="%.0e",
            disabled=disabled,
        )
    with c2:
        allow_cell = st.selectbox(
            "Relax cell?", ["Yes (recommended)", "No (fixed cell)"],
            index=0, disabled=disabled
        ).startswith("Yes")

    run_btn = st.button("Run Elastic Workflow", type="primary", disabled=disabled)

    # Show last results if available
    if st.session_state.elastic_payload and not st.session_state.elastic_running:
        _render_payload(st.session_state.elastic_payload)
        with st.expander("Debug info", expanded=False):
            st.json(st.session_state.elastic_debug or {})

    if not run_btn or st.session_state.elastic_running:
        return

    # === run once ===
    st.session_state.elastic_running = True
    try:
        status = st.status("Building flow…", expanded=True)
        status.write("Preparing makers…")

        bulk_relax = MACERelaxMaker(relax_cell=True,  relax_kwargs={"fmax": float(fmax)})
        el_relax   = MACERelaxMaker(relax_cell=bool(allow_cell), relax_kwargs={"fmax": float(fmax)})

        maker = ElasticMaker(bulk_relax_maker=bulk_relax, elastic_relax_maker=el_relax)
        flow = maker.make(structure=pmg_obj)

        status.update(label="Running locally…", state="running")
        rs = run_locally(flow, create_folders=True, ensure_success=True)

        # Quick job table
        rows = []
        items = list(rs.items()) if isinstance(rs, dict) else list(enumerate(rs))
        for j, r in items:
            nm = (getattr(r, "name", None) or str(j))
            rows.append({"job": str(j)[:8], "name": nm[:80], "error": bool(getattr(r, "error", None))})
        st.dataframe(rows, use_container_width=True)

        status.update(label="Extracting results…", state="running")
        C, P, meta, resolved_plain, samples = _extract_elastic(rs, flow)

        # Backfill E, ν if missing
        K, G = P.get("k_vrh"), P.get("g_vrh")
        if isinstance(K, (int, float)) and isinstance(G, (int, float)):
            denom = 3 * K + G
            if denom:
                P.setdefault("y_mod", 9 * K * G / denom)
                P.setdefault("homogeneous_poisson", (3 * K - 2 * G) / (2 * denom))

        payload = {"C_GPa": np.array(C, dtype=float).tolist(), "props": P}
        st.session_state.elastic_payload = payload
        st.session_state.elastic_debug = {"found_in": meta.get("found_in"), "jobs_seen": meta.get("jobs_seen")}

        status.update(label="Done ✅", state="complete")
        _render_payload(payload)

        # Show sanitized raw objects (helps future debugging but safe when success)
        with st.expander("Resolved flow.output (sanitized)", expanded=False):
            st.json(_to_plain(resolved_plain))
        if samples:
            with st.expander("Sample job outputs (sanitized)", expanded=False):
                st.json(_to_plain(samples))

    except Exception as e:
        st.error(f"Elastic workflow failed: {e}")
        # Dump small sanitized samples so we can see where results live
        with st.expander("Resolved flow.output (sanitized)", expanded=True):
            try:
                resolved = flow.output.resolve(rs)  # may fail; that's fine
                st.json(_to_plain(resolved))
            except Exception as ee:
                st.write(f"(Could not resolve flow.output: {ee})")
        with st.expander("Sample job outputs (sanitized)", expanded=True):
            dump = []
            items = list(rs.items()) if isinstance(rs, dict) else list(enumerate(rs))
            for i, (j, r) in enumerate(items):
                if i >= 5:
                    break
                dump.append({"job": str(j), "name": getattr(r, "name", None), "output": _to_plain(getattr(r, "output", None))})
            st.json(dump)
        with st.expander("Exception", expanded=False):
            st.exception(e)
    finally:
        st.session_state.elastic_running = False
