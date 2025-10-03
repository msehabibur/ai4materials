from __future__ import annotations
import io
import json
import csv
from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import streamlit as st
from pymatgen.core import Structure

from monty.json import jsanitize  # <-- key: turn typed objects into plain dicts
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import MACERelaxMaker
from jobflow import run_locally


# ---------- utilities ----------
def _to_gpa(val: Any):
    """Convert Pa->GPa if values look like Pascals; pass through lists/dicts."""
    if val is None:
        return None
    if isinstance(val, dict):
        return {k: _to_gpa(v) for k, v in val.items()}
    arr = np.array(val, dtype=float)
    if np.nanmax(np.abs(arr)) > 1e5:  # likely Pascals
        arr = arr * 1e-9
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def _pack_csv(C: np.ndarray, props: dict) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Elastic tensor C (GPa) — 6x6 IEEE"])
    for row in C:
        w.writerow([f"{v:.6f}" for v in row])
    w.writerow([])
    w.writerow(["Property", "Value"])
    keys = [
        "k_voigt", "k_reuss", "k_vrh",
        "g_voigt", "g_reuss", "g_vrh",
        "y_mod", "homogeneous_poisson",
        "universal_anisotropy",
    ]
    for k in keys:
        v = props.get(k, None)
        w.writerow([k, f"{v:.6f}" if isinstance(v, (int, float)) else "—"])
    return buf.getvalue().encode()


# ---------- robust extractors ----------
def _maybe_tensor_from_plain_dict(d: dict) -> Optional[Tuple[List[List[float]], dict]]:
    """
    Try common shapes used by atomate2 elastic flows across versions.
    Return (C_ij_GPa, props_dict) if found.
    """
    # 1) {"elastic_tensor": {"ieee_format": 6x6}, "derived_properties": {...}}
    if "elastic_tensor" in d and isinstance(d["elastic_tensor"], dict):
        et = d["elastic_tensor"]
        mat = et.get("ieee_format") or et.get("matrix") or et.get("C_ij") or None
        if mat is not None:
            props = d.get("derived_properties", {}) or {}
            return _to_gpa(mat), _to_gpa(props)

    # 2) Minimal: {"C_ij": ..., "properties": {...}}
    if all(k in d for k in ("C_ij", "properties")):
        return _to_gpa(d["C_ij"]), _to_gpa(d["properties"])

    # 3) Some flows tuck results under "output"/"data"/"result"/"results"
    for k in ("output", "data", "result", "results"):
        v = d.get(k)
        if isinstance(v, dict):
            out = _maybe_tensor_from_plain_dict(v)
            if out:
                return out
    return None


def _to_plain(obj: Any) -> Any:
    """
    Convert any monty-serializable/typed object to plain JSON-friendly Python types.
    """
    try:
        return jsanitize(obj, strict=False)
    except Exception:
        # Best effort fallback
        if hasattr(obj, "as_dict"):
            try:
                return jsanitize(obj.as_dict(), strict=False)
            except Exception:
                return obj
        return obj


def _walk_plain_for_tensor(obj: Any) -> Optional[Tuple[List[List[float]], dict]]:
    """Recursively walk any plain structure looking for an elastic tensor + props."""
    if isinstance(obj, dict):
        got = _maybe_tensor_from_plain_dict(obj)
        if got:
            return got
        for v in obj.values():
            hit = _walk_plain_for_tensor(v)
            if hit:
                return hit
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            hit = _walk_plain_for_tensor(v)
            if hit:
                return hit
    return None


def _extract_elastic(rs: Dict[str, Any], flow_obj: Any) -> Tuple[np.ndarray, dict, dict]:
    """
    Resolve flow.output first (jobflow references), sanitize to plain dicts,
    then fall back to sanitized job outputs.
    Returns (C_ij_GPa ndarray, props dict, debug_meta dict)
    """
    debug_meta = {"jobs_seen": [], "found_in": None}

    # Pass 0: try resolving flow.output (most reliable)
    try:
        resolved = flow_obj.output.resolve(rs)
        resolved_plain = _to_plain(resolved)
        got = _walk_plain_for_tensor(resolved_plain)
        if got:
            C, P = got
            debug_meta["found_in"] = {"scope": "flow.output.resolve(rs)"}
            return np.array(C, dtype=float), P or {}, debug_meta
    except Exception:
        pass  # continue

    # Pass 1: sanitize per-job output & metadata
    for jid, rec in rs.items():
        debug_meta["jobs_seen"].append({"id": str(jid), "name": getattr(rec, "name", None)})
        for field in ("output", "metadata"):
            obj = getattr(rec, field, None)
            if obj is None:
                continue
            plain = _to_plain(obj)
            got = _walk_plain_for_tensor(plain)
            if got:
                C, P = got
                debug_meta["found_in"] = {"job_id": str(jid), "field": field}
                return np.array(C, dtype=float), P or {}, debug_meta

    # Pass 2: look for a job named like 'fit_elastic_tensor' explicitly
    for jid, rec in rs.items():
        name = (getattr(rec, "name", "") or "").lower()
        if "fit_elastic" in name or "elastic" in name:
            plain = _to_plain(getattr(rec, "output", None))
            got = _walk_plain_for_tensor(plain)
            if got:
                C, P = got
                debug_meta["found_in"] = {"job_id": str(jid), "field": "output(name-match)"}
                return np.array(C, dtype=float), P or {}, debug_meta

    # Pass 3: sanitize whole responses & deep-scan as last resort
    whole_plain = _to_plain(rs)
    got = _walk_plain_for_tensor(whole_plain)
    if got:
        C, P = got
        debug_meta["found_in"] = {"scope": "responses-deep-scan"}
        return np.array(C, dtype=float), P or {}, debug_meta

    raise RuntimeError("Elastic results not found in job outputs.")


# ---------- rendering ----------
def _render_payload(payload: dict):
    C = np.array(payload["C_GPa"], dtype=float)
    P = payload["props"]

    st.markdown("**Elastic tensor C (GPa, 6×6 IEEE)**")
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
    with d1:
        st.download_button("⬇️ JSON", json_bytes, "elastic_results.json")
    with d2:
        st.download_button("⬇️ CSV", csv_bytes, "elastic_results.csv")


# ---------- main tab ----------
def elastic_tab(pmg_obj: Structure | None):
    st.subheader("🧱 Elastic — default flow (MACE)")
    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer first.")
        return

    # Keep state across reruns
    if "elastic_running" not in st.session_state:
        st.session_state.elastic_running = False
    if "elastic_payload" not in st.session_state:
        st.session_state.elastic_payload = None
    if "elastic_debug" not in st.session_state:
        st.session_state.elastic_debug = None

    disabled = st.session_state.elastic_running

    c1, c2 = st.columns(2)
    with c1:
        # flexible sci-notation range
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

    # === Atomic run ===
    st.session_state.elastic_running = True
    try:
        status = st.status("Building flow…", expanded=True)
        status.write("Preparing makers…")

        # Version-agnostic: use default MACE-backed relaxers
        bulk_relax = MACERelaxMaker(
            relax_cell=True,
            relax_kwargs={"fmax": float(fmax)},
        )
        el_relax = MACERelaxMaker(
            relax_cell=bool(allow_cell),
            relax_kwargs={"fmax": float(fmax)},
        )

        maker = ElasticMaker(
            bulk_relax_maker=bulk_relax,
            elastic_relax_maker=el_relax,
        )

        flow = maker.make(structure=pmg_obj)
        status.update(label="Running locally…", state="running")

        rs = run_locally(
            flow,
            create_folders=True,
            ensure_success=True
        )

        # Small job table for sanity (names can be UUID-like in some versions)
        rows = []
        for j, r in rs.items():
            nm = (getattr(r, "name", None) or str(j))
            rows.append({"job": str(j)[:8], "name": nm[:80], "error": bool(getattr(r, "error", None))})
        st.dataframe(rows, use_container_width=True)

        status.update(label="Extracting results…", state="running")
        C, P, meta = _extract_elastic(rs, flow)

        # Backfill E, ν if missing (rare)
        K, G = P.get("k_vrh"), P.get("g_vrh")
        if isinstance(K, (int, float)) and isinstance(G, (int, float)):
            denom = 3 * K + G
            if denom:
                E = 9 * K * G / denom
                nu = (3 * K - 2 * G) / (2 * denom)
                P.setdefault("y_mod", E)
                P.setdefault("homogeneous_poisson", nu)

        payload = {"C_GPa": C.tolist(), "props": P}
        st.session_state.elastic_payload = payload
        st.session_state.elastic_debug = {
            "found_in": meta.get("found_in"),
            "jobs_seen": meta.get("jobs_seen"),
        }

        status.update(label="Done ✅", state="complete")
        _render_payload(payload)

    except Exception as e:
        st.error(f"Elastic workflow failed: {e}")
        with st.expander("Debug dump (last run)", expanded=False):
            try:
                st.exception(e)
            except Exception:
                st.write(str(e))
    finally:
        st.session_state.elastic_running = False
