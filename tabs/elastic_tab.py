# tabs/elastic_tab.py
from __future__ import annotations

import json
import numbers
import numpy as np
import streamlit as st
from monty.json import jsanitize
from pymatgen.core import Structure

from jobflow import run_locally, SETTINGS
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import MACERelaxMaker


# -------------------- tiny helpers --------------------
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
    """Heuristically convert Pa→GPa if needed, return ndarray in GPa."""
    arr = np.array(x, dtype=float)
    try:
        if np.nanmax(np.abs(arr)) > 1e5:  # looks like Pascals
            arr *= 1e-9
    except Exception:
        pass
    return arr


def _props_to_gpa(props: dict) -> dict:
    """Convert known scalar props Pa→GPa when needed."""
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
            # prefer likely keys first
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


def _render_results_json(C_gpa: np.ndarray, P: dict):
    """Display clean JSON and a JSON download only (no tables/metrics)."""
    payload = {
        "elastic_tensor_GPa": np.array(C_gpa, dtype=float).tolist(),  # 6x6
        "derived_properties_GPa": {
            k: (float(v) if isinstance(v, (int, float)) else v) for k, v in P.items()
        },
    }
    st.json(payload)
    st.download_button(
        "⬇️ Download results (JSON)",
        json.dumps(payload, indent=2).encode("utf-8"),
        "elastic_results.json",
        mime="application/json",
    )


# -------------------- Streamlit tab --------------------
def elastic_tab(pmg_obj: Structure | None):
    st.subheader("🧱 Elastic — MACE (JSON output only)")
    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer first.")
        return

    st.session_state.setdefault("elastic_running", False)
    st.session_state.setdefault("elastic_payload", None)

    # Optional: flip on only when you want internal details
    show_debug = st.toggle("Show debug details", value=False)

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

    # Show last results (JSON-only)
    if (not st.session_state["elastic_running"]) and st.session_state["elastic_payload"]:
        C_gpa = np.array(st.session_state["elastic_payload"]["C_GPa"], dtype=float)
        P     = st.session_state["elastic_payload"]["props"]
        _render_results_json(C_gpa, P)

    if (not run_btn) or st.session_state["elastic_running"]:
        return

    # ===== Run once =====
    st.session_state["elastic_running"] = True
    try:
        if show_debug:
            st.info("Building MACE elastic flow…")

        bulk_relax = MACERelaxMaker(relax_cell=True,  relax_kwargs={"fmax": float(fmax)})
        el_relax   = MACERelaxMaker(relax_cell=bool(allow_cell), relax_kwargs={"fmax": float(fmax)})
        maker = ElasticMaker(bulk_relax_maker=bulk_relax, elastic_relax_maker=el_relax)
        flow = maker.make(structure=pmg_obj)

        if show_debug:
            st.info("Running locally…")

        # Follow the working pattern: do NOT pass store=; use SETTINGS.JOB_STORE afterwards
        rs = run_locally(flow, create_folders=True, ensure_success=False)

        # Show failing jobs only when debug is on
        if show_debug:
            items = list(rs.items()) if isinstance(rs, dict) else list(enumerate(rs))
            rows = []
            for j, r in items:
                nm = getattr(r, "name", None) or str(j)
                rows.append({"job": str(j)[:8], "name": nm[:80], "error": bool(getattr(r, "error", None))})
            st.dataframe(rows, use_container_width=True)
            failed = [(j, r) for j, r in items if getattr(r, "error", None)]
            if failed:
                st.error(f"{len(failed)} job(s) failed.")
                for j, r in failed[:5]:
                    with st.expander(f"❌ {getattr(r,'name',str(j))} — job {str(j)}"):
                        if getattr(r, "error", None):
                            st.exception(getattr(r, "error"))
                        st.caption("Output (sanitized)")
                        st.json(_to_plain(getattr(r, "output", None)))
                        st.caption("Metadata (sanitized)")
                        st.json(_to_plain(getattr(r, "metadata", None)))
                st.stop()

        # Extract results from the store like in your working script
        store = SETTINGS.JOB_STORE
        if store is None:
            raise RuntimeError("SETTINGS.JOB_STORE is None after run_locally; cannot query results.")
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
            # Fallback: deep-scan responses to find 6×6
            plain = _to_plain(rs)
            C_raw = _find_tensor_anywhere(plain)
            if C_raw is None:
                raise RuntimeError("No 'fit_elastic_tensor' document found and no 6×6 tensor in responses.")
            P = {}
        else:
            out = doc.get("output", {}) if isinstance(doc, dict) else {}
            et  = out.get("elastic_tensor", {}) if isinstance(out, dict) else {}
            C_raw = _pick_c_6x6(et)
            if C_raw is None:
                raise RuntimeError("Elastic tensor found, but no 6×6 field among ieee_format/voigt/matrix/C_ij/C.")
            P = out.get("derived_properties", {}) or {}

        # ---- Normalize units and compute E, ν in GPa ----
        C_gpa = _as_gpa_array(C_raw)   # tensor → GPa
        P     = _props_to_gpa(P)       # props → GPa if needed

        K = P.get("k_vrh")
        G = P.get("g_vrh")
        if isinstance(K, (int, float)) and isinstance(G, (int, float)):
            denom = 3.0 * K + G
            if denom != 0.0:
                P["y_mod"] = 9.0 * K * G / denom               # E (GPa)
                P["homogeneous_poisson"] = (3.0 * K - 2.0 * G) / (2.0 * denom)
        else:
            # if y_mod was in Pa, convert to GPa
            y = P.get("y_mod")
            if isinstance(y, (int, float)) and abs(y) > 1e5:
                P["y_mod"] = y * 1e-9

        # Persist + show JSON-only
        st.session_state["elastic_payload"] = {"C_GPa": C_gpa.tolist(), "props": P}
        _render_results_json(C_gpa, P)

    except Exception as e:
        st.error(f"Elastic workflow failed: {e}")
        with st.expander("Exception", expanded=True):
            st.exception(e)
    finally:
        st.session_state["elastic_running"] = False
