# tabs/elastic_tab.py
from __future__ import annotations

import json
import numbers
import threading
import time
from typing import List, Optional

import numpy as np
import streamlit as st
from monty.json import jsanitize
from pymatgen.core import Structure

from jobflow import run_locally, SETTINGS
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import MACERelaxMaker

# ==================== helpers ====================
def _to_plain(obj):
    try:
        return jsanitize(obj, strict=False)
    except Exception:
        if hasattr(obj, "as_dict"):
            try:
                return jsanitize(obj.as_dict(), strict=False)
            except Exception:
                return obj
        return obj


def _looks_like_6x6(mat):
    if isinstance(mat, (list, tuple)) and len(mat) == 6:
        for row in mat:
            if not isinstance(row, (list, tuple)) or len(row) != 6:
                return False
            for x in row:
                if not isinstance(x, numbers.Number):
                    return False
        return True
    return False


def _as_gpa_array(x):
    """Convert Pa→GPa if needed, return ndarray in GPa."""
    arr = np.array(x, dtype=float)
    try:
        if np.nanmax(np.abs(arr)) > 1e5:  # likely Pascals
            arr *= 1e-9
    except Exception:
        pass
    return arr


def _props_to_gpa(props: dict) -> dict:
    """Scalar props Pa→GPa when needed."""
    if not isinstance(props, dict):
        return {}
    out = {}
    for k, v in props.items():
        if isinstance(v, (int, float)):
            vv = float(v)
            if abs(vv) > 1e5:  # likely Pa
                vv *= 1e-9
            out[k] = vv
        else:
            out[k] = v
    return out


def _pick_c_6x6(elastic_tensor_obj: dict):
    """Choose a 6×6 matrix from common field names."""
    if not isinstance(elastic_tensor_obj, dict):
        return None
    for key in ("ieee_format", "voigt", "matrix", "C_ij", "C"):
        if key in elastic_tensor_obj and _looks_like_6x6(elastic_tensor_obj[key]):
            return elastic_tensor_obj[key]
    return None


def _find_tensor_anywhere(d):
    """Fallback deep-search for a 6×6 matrix in nested dict/list."""
    def walk(x):
        if _looks_like_6x6(x):
            return x
        if isinstance(x, dict):
            # try likely keys first
            for k in ("elastic_tensor", "tensor", "C_ij", "C", "voigt", "ieee_format",
                      "output", "data", "results", "result", "metadata"):
                if k in x:
                    got = walk(x[k])
                    if got is not None:
                        return got
            for v in x.values():
                got = walk(v)
                if got is not None:
                    return got
        elif isinstance(x, (list, tuple)):
            for v in x:
                got = walk(v)
                if got is not None:
                    return got
        return None
    return walk(d)


def _format_list_of_lists(arr):
    """Pretty print a 6×6 list-of-lists as Python-style text."""
    A = np.array(arr, dtype=float)
    lines = ["["]
    for i, row in enumerate(A):
        row_txt = ", ".join(f"{v:.6f}" for v in row)
        if i < len(A) - 1:
            lines.append(f"  [{row_txt}],")
        else:
            lines.append(f"  [{row_txt}]")
    lines.append("]")
    return "\n".join(lines)


def _render_list_style(C_gpa: np.ndarray, P: dict):
    """Show C_ij and properties as list-style text (no tables/heatmaps)."""
    st.markdown("### Elastic results (list style)")
    st.markdown("**C_ij (GPa), 6×6 (Voigt/IEEE)**")
    st.code(_format_list_of_lists(C_gpa), language="python")

    # Bullet list of key properties
    k = P.get("k_vrh"); g = P.get("g_vrh"); e = P.get("y_mod")
    nu = P.get("homogeneous_poisson")
    au = P.get("universal_anisotropy")

    bullets = []
    bullets.append(f"- K_VRH (GPa): {k:.3f}" if isinstance(k, (int, float)) else "- K_VRH (GPa): —")
    bullets.append(f"- G_VRH (GPa): {g:.3f}" if isinstance(g, (int, float)) else "- G_VRH (GPa): —")
    bullets.append(f"- E (GPa): {e:.3f}" if isinstance(e, (int, float)) else "- E (GPa): —")
    bullets.append(f"- Poisson ν: {nu:.3f}" if isinstance(nu, (int, float)) else "- Poisson ν: —")
    bullets.append(f"- Anisotropy AU: {au:.3f}" if isinstance(au, (int, float)) else "- Anisotropy AU: —")

    st.markdown("\n".join(bullets))

    # Also provide a JSON download for scripting if needed
    payload = {
        "elastic_tensor_GPa": np.array(C_gpa, dtype=float).tolist(),
        "derived_properties_GPa": {
            k2: (float(v2) if isinstance(v2, (int, float)) else v2) for k2, v2 in P.items()
        },
    }
    st.download_button(
        "⬇️ Download results (JSON)",
        json.dumps(payload, indent=2).encode("utf-8"),
        "elastic_results.json",
        mime="application/json",
    )


# ==================== progress utilities ====================
def _collect_job_uuids(flow) -> List[str]:
    """Try to collect all job UUIDs from a flow before submission."""
    uuids = []
    try:
        for j in flow.jobs:
            if getattr(j, "uuid", None):
                uuids.append(j.uuid)
    except Exception:
        pass
    return uuids


def _count_finished_jobs(store, uuids: List[str]) -> Optional[int]:
    """Return number of finished jobs among uuids; None if querying fails."""
    try:
        store.connect()
    except Exception:
        # continue; many stores don't need explicit connect
        pass

    try:
        # Prefer "state" if present; fall back to presence of completed_at
        finished = 0
        for u in uuids:
            doc = store.query_one(
                {"uuid": u},
                properties=["state", "completed_at"],
                load=False,
                sort={"completed_at": -1},
            )
            if not doc:
                continue
            state = doc.get("state")
            if state in ("COMPLETED", "FAILED"):
                finished += 1
            elif doc.get("completed_at"):
                finished += 1
        return finished
    except Exception:
        return None


def _run_flow_worker(flow, kwargs: dict):
    """Worker thread to execute the flow."""
    try:
        rs = run_locally(flow, **kwargs)
        st.session_state["elastic_rs_plain"] = _to_plain(rs)
        st.session_state["elastic_error"] = None
    except Exception as e:
        st.session_state["elastic_error"] = e
    finally:
        st.session_state["elastic_running"] = False


# ==================== main Streamlit tab ====================
def elastic_tab(pmg_obj: Structure | None):
    st.subheader("🧱 Elastic — MACE (list output)")

    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer first.")
        return

    # session state keys
    st.session_state.setdefault("elastic_running", False)
    st.session_state.setdefault("elastic_payload", None)
    st.session_state.setdefault("elastic_job_uuids", [])
    st.session_state.setdefault("elastic_total_jobs", 0)
    st.session_state.setdefault("elastic_rs_plain", None)
    st.session_state.setdefault("elastic_error", None)

    disabled = st.session_state["elastic_running"]
    c1, c2 = st.columns(2)
    with c1:
        fmax = st.number_input(
            "Relax fmax (eV/Å)",
            min_value=1e-8, max_value=1e-1,
            value=1e-5, step=1e-8, format="%.0e",
            disabled=disabled,
        )
    with c2:
        allow_cell = st.selectbox(
            "Relax cell?",
            ["Yes (recommended)", "No (fixed cell)"],
            index=0, disabled=disabled
        ).startswith("Yes")

    run_btn = st.button("Run Elastic Workflow", type="primary", disabled=disabled)

    # Show any cached results
    if (not st.session_state["elastic_running"]) and st.session_state["elastic_payload"]:
        C_gpa = np.array(st.session_state["elastic_payload"]["C_GPa"], dtype=float)
        P     = st.session_state["elastic_payload"]["props"]
        _render_list_style(C_gpa, P)

    # ===== Kick off run (once) =====
    if run_btn and (not st.session_state["elastic_running"]):
        st.session_state["elastic_running"] = True
        st.session_state["elastic_rs_plain"] = None
        st.session_state["elastic_error"] = None

        # Build flow
        bulk_relax = MACERelaxMaker(relax_cell=True,  relax_kwargs={"fmax": float(fmax)})
        el_relax   = MACERelaxMaker(relax_cell=bool(allow_cell), relax_kwargs={"fmax": float(fmax)})
        maker = ElasticMaker(bulk_relax_maker=bulk_relax, elastic_relax_maker=el_relax)
        flow = maker.make(structure=pmg_obj)

        # collect uuids for progress tracking
        uuids = _collect_job_uuids(flow)
        st.session_state["elastic_job_uuids"] = uuids
        st.session_state["elastic_total_jobs"] = len(uuids) if uuids else 0

        # launch worker thread
        t = threading.Thread(
            target=_run_flow_worker,
            args=(flow, dict(create_folders=True, ensure_success=False)),
            daemon=True,
        )
        t.start()

    # ===== While running: show progress =====
    if st.session_state["elastic_running"]:
        store = SETTINGS.JOB_STORE
        progress_area = st.empty()
        note_area = st.empty()

        # If we know total jobs, use a true progress bar; else spinner fallback.
        total = st.session_state.get("elastic_total_jobs", 0)
        if total:
            pb = progress_area.progress(0, text="Submitting jobs…")
            last = -1
            while st.session_state["elastic_running"]:
                finished = _count_finished_jobs(store, st.session_state["elastic_job_uuids"])
                if finished is None:
                    # store can't be queried -> fallback to spinner
                    break
                finished = int(finished)
                if finished != last:
                    frac = max(0.0, min(1.0, finished / total))
                    pb.progress(int(frac * 100), text=f"Running elastic workflow… {finished}/{total} finished")
                    note_area.info("Tracking job completion from the JobStore…")
                    last = finished
                time.sleep(1.0)
            progress_area.empty()
            note_area.empty()
        else:
            # Unknown job count; show an indeterminate spinner
            with st.spinner("Running elastic workflow…"):
                while st.session_state["elastic_running"]:
                    time.sleep(1.0)

        # When the worker finishes, continue below to extract results.

    # ===== After run finishes: extract & render =====
    if (not st.session_state["elastic_running"]) and (st.session_state["elastic_rs_plain"] or st.session_state["elastic_error"]):
        if st.session_state["elastic_error"] is not None:
            e = st.session_state["elastic_error"]
            st.error(f"Elastic workflow failed: {e}")
            with st.expander("Exception", expanded=True):
                st.exception(e)
            return

        try:
            # Prefer querying the 'fit_elastic_tensor' document
            store = SETTINGS.JOB_STORE
            if store is None:
                raise RuntimeError("SETTINGS.JOB_STORE is None after run; cannot query results.")
            try:
                store.connect()
            except Exception:
                pass

            doc = store.query_one(
                {"name": "fit_elastic_tensor"},
                properties=["output.elastic_tensor", "output.derived_properties"],
                load=True,
                sort={"completed_at": -1},
            )

            if doc is None:
                # fallback: deep-scan responses
                plain = st.session_state["elastic_rs_plain"]
                C_raw = _find_tensor_anywhere(plain)
                if C_raw is None:
                    raise RuntimeError("No 'fit_elastic_tensor' document and no 6×6 tensor in responses.")
                P = {}
            else:
                out = doc.get("output", {}) if isinstance(doc, dict) else {}
                et  = out.get("elastic_tensor", {}) if isinstance(out, dict) else {}
                C_raw = _pick_c_6x6(et)
                if C_raw is None:
                    raise RuntimeError("Elastic tensor found, but no 6×6 field among ieee_format/voigt/matrix/C_ij/C.")
                P = out.get("derived_properties", {}) or {}

            # Units normalization & derived values in GPa
            C_gpa = _as_gpa_array(C_raw)
            P     = _props_to_gpa(P)

            K = P.get("k_vrh")
            G = P.get("g_vrh")
            if isinstance(K, (int, float)) and isinstance(G, (int, float)):
                denom = 3.0 * K + G
                if denom != 0.0:
                    P["y_mod"] = 9.0 * K * G / denom
                    P["homogeneous_poisson"] = (3.0 * K - 2.0 * G) / (2.0 * denom)
            else:
                y = P.get("y_mod")
                if isinstance(y, (int, float)) and abs(y) > 1e5:
                    P["y_mod"] = y * 1e-9

            st.session_state["elastic_payload"] = {"C_GPa": C_gpa.tolist(), "props": P}
            _render_list_style(C_gpa, P)

        except Exception as e:
            st.error(f"Elastic workflow finished but result extraction failed: {e}")
            with st.expander("Exception", expanded=True):
                st.exception(e)
