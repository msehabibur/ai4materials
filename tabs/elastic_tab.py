from __future__ import annotations
import io
import json
import csv
from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import streamlit as st
from pymatgen.core import Structure

from monty.json import jsanitize
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import MACERelaxMaker
from jobflow import run_locally


# ---------- utils ----------
def _to_gpa(val: Any):
    if val is None:
        return None
    if isinstance(val, dict):
        return {k: _to_gpa(v) for k, v in val.items()}
    arr = np.array(val, dtype=float)
    # If it looks like Pascals, convert to GPa
    if np.nanmax(np.abs(arr)) > 1e5:
        arr = arr * 1e-9
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def _pack_csv(C: np.ndarray, props: dict) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Elastic tensor C (GPa) — 6x6 (Voigt/IEEE order)"])
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


def _to_plain(obj: Any) -> Any:
    try:
        return jsanitize(obj, strict=False)
    except Exception:
        if hasattr(obj, "as_dict"):
            try:
                return jsanitize(obj.as_dict(), strict=False)
            except Exception:
                return obj
        return obj


# ---------- robust extraction ----------
def _extract_from_plain_dict(d: dict) -> Optional[Tuple[List[List[float]], dict]]:
    """
    Try a broad set of field patterns seen across atomate2 versions.
    Return (C_ij_GPa, props_dict) if found.
    """
    if not isinstance(d, dict):
        return None

    # A) Canonical modern shape
    if "elastic_tensor" in d and isinstance(d["elastic_tensor"], dict):
        et = d["elastic_tensor"]
        for key in ("ieee_format", "voigt", "matrix", "C_ij", "C"):
            if key in et:
                props = d.get("derived_properties", {}) or d.get("properties", {}) or {}
                return _to_gpa(et[key]), _to_gpa(props)

    # B) Flat / older shapes
    for key in ("C_ij", "C", "tensor", "elastic", "elastic_voigt"):
        if key in d:
            C = d[key]
            props = d.get("derived_properties", {}) or d.get("properties", {}) or {}
            return _to_gpa(C), _to_gpa(props)

    # C) Nested under common containers
    for k in ("output", "data", "result", "results", "elastic_data", "elastic"):
        v = d.get(k)
        if isinstance(v, dict):
            got = _extract_from_plain_dict(v)
            if got:
                return got

    return None


def _walk_any(obj: Any) -> Optional[Tuple[List[List[float]], dict]]:
    if isinstance(obj, dict):
        got = _extract_from_plain_dict(obj)
        if got:
            return got
        for v in obj.values():
            hit = _walk_any(v)
            if hit:
                return hit
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            hit = _walk_any(v)
            if hit:
                return hit
    return None


def _extract_elastic(rs: Dict[str, Any], flow_obj: Any) -> Tuple[np.ndarray, dict, dict, dict, list]:
    """
    Resolve flow.output, sanitize, and try extraction; then fall back to per-job outputs/metadata.
    Returns:
      C (np.ndarray), P (dict), meta (dict),
      resolved_plain (dict|list), samples (list of first N sanitized job outputs)
    """
    debug_meta = {"found_in": None, "jobs_seen": []}

    # Pass 0: resolve flow.output
    resolved_plain = None
    try:
        resolved = flow_obj.output.resolve(rs)
        resolved_plain = _to_plain(resolved)
        got = _walk_any(resolved_plain)
        if got:
            C, P = got
            debug_meta["found_in"] = {"scope": "flow.output.resolve(rs)"}
            return np.array(C, dtype=float), (P or {}), debug_meta, resolved_plain, []
    except Exception:
        pass

    # Pass 1: scan jobs
    samples = []
    for i, (jid, rec) in enumerate(rs.items()):
        debug_meta["jobs_seen"].append({"id": str(jid), "name": getattr(rec, "name", None)})
        for field in ("output", "metadata"):
            obj = getattr(rec, field, None)
            if obj is None:
                continue
            plain = _to_plain(obj)
            if i < 5 and field == "output":  # collect a few samples for UI
                samples.append({"job_id": str(jid), "name": getattr(rec, "name", None), "output_sample": plain})
            got = _walk_any(plain)
            if got:
                C, P = got
                debug_meta["found_in"] = {"job_id": str(jid), "field": field}
                return np.array(C, dtype=float), (P or {}), debug_meta, resolved_plain, samples

    # Pass 2: deep-scan whole responses
    whole_plain = _to_plain(rs)
    got = _walk_any(whole_plain)
    if got:
        C, P = got
        debug_meta["found_in"] = {"scope": "responses-deep-scan"}
        return np.array(C, dtype=float), (P or {}), debug_meta, resolved_plain, samples

    raise RuntimeError("Elastic results not found in job outputs.")


# ---------- render ----------
def _render_payload(payload: dict):
    C = np.array(payload["C_GPa"], dtype=float)
    P = payload["props"]

    st.markdown("**Elastic tensor C (GPa, 6×6 IEEE/Voigt)**")
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
    st.subheader("🧱 Elastic)")
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

        # quick job table
        rows = []
        for j, r in rs.items():
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

        # Always show sanitized raw objects to aid diagnostics (harmless when success)
        with st.expander("Resolved flow.output (sanitized)", expanded=False):
            st.json(_to_plain(resolved_plain))
        if samples:
            with st.expander("Sample job outputs (sanitized)", expanded=False):
                st.json(_to_plain(samples))

    except Exception as e:
        st.error(f"Elastic workflow failed: {e}")
        # Show raw data to see where it lives
        with st.expander("Resolved flow.output (sanitized)", expanded=True):
            try:
                resolved = flow.output.resolve(rs)
                st.json(_to_plain(resolved))
            except Exception as ee:
                st.write(f"(Could not resolve flow.output: {ee})")
        with st.expander("Sample job outputs (sanitized)", expanded=True):
            # dump first few outputs
            dump = []
            for i, (j, r) in enumerate(rs.items()):
                if i >= 5:
                    break
                dump.append({"job": str(j), "name": getattr(r, "name", None), "output": _to_plain(getattr(r, "output", None))})
            st.json(dump)
        with st.expander("Exception", expanded=False):
            st.exception(e)
    finally:
        st.session_state.elastic_running = False
