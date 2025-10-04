# tabs/elastic_tab.py
from __future__ import annotations

import os, json, csv, glob, numbers
import numpy as np
import streamlit as st
from monty.json import jsanitize
from pymatgen.core import Structure

from jobflow import run_locally, SETTINGS
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import MACERelaxMaker

# -------------------- small helpers --------------------
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
        try:
            for row in mat:
                if not isinstance(row, (list, tuple)) or len(row) != 6:
                    return False
                for x in row:
                    if not isinstance(x, numbers.Number):
                        return False
            return True
        except Exception:
            return False
    return False

def _to_gpa(x):
    arr = np.array(x, dtype=float)
    try:
        if np.nanmax(np.abs(arr)) > 1.0e5:  # looks like Pa -> convert to GPa
            arr = arr * 1.0e-9
    except Exception:
        pass
    return arr

def _pack_csv(C, props):
    import io, csv as _csv
    sio = io.StringIO()
    w = _csv.writer(sio)
    w.writerow(["Elastic tensor C (GPa), Voigt/IEEE 6×6"])
    for row in C:
        w.writerow([f"{v:.6f}" for v in row])
    w.writerow([])
    w.writerow(["Property","Value"])
    for k in [
        "k_voigt","k_reuss","k_vrh",
        "g_voigt","g_reuss","g_vrh",
        "y_mod","homogeneous_poisson","universal_anisotropy",
    ]:
        v = props.get(k, None)
        w.writerow([k, f"{v:.6f}" if isinstance(v, (int,float)) else "—"])
    return sio.getvalue().encode("utf-8")

# -------------------- Streamlit tab --------------------
def elastic_tab(pmg_obj: Structure | None):
    st.subheader("🧱 Elastic — MACE default workflow (Jobflow query-based extraction)")

    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer first.")
        return

    st.session_state.setdefault("elastic_running", False)
    st.session_state.setdefault("elastic_payload", None)

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

    # Show last results
    payload = st.session_state["elastic_payload"]
    if (not st.session_state["elastic_running"]) and payload:
        _render_results(payload)

    if (not run_btn) or st.session_state["elastic_running"]:
        return

    # ====== Run once ======
    st.session_state["elastic_running"] = True
    try:
        status = st.status("Building flow…", expanded=True)
        status.write("Preparing makers…")

        # Build the exact makers you showed in your example
        bulk_relax = MACERelaxMaker(
            relax_cell=True,
            relax_kwargs={"fmax": float(fmax)}
        )
        el_relax = MACERelaxMaker(
            relax_cell=bool(allow_cell),
            relax_kwargs={"fmax": float(fmax)}
        )
        maker = ElasticMaker(
            bulk_relax_maker=bulk_relax,
            elastic_relax_maker=el_relax,
        )
        flow = maker.make(structure=pmg_obj)

        status.update(label="Running locally…", state="running")

        # CRUCIAL: follow your pattern — do NOT pass store=, let jobflow set SETTINGS.JOB_STORE
        rs = run_locally(flow, create_folders=True, ensure_success=False)

        # Show quick job table + any failures (so you see why)
        rows = []
        items = list(rs.items()) if isinstance(rs, dict) else list(enumerate(rs))
        for j, r in items:
            nm = getattr(r, "name", None) or str(j)
            rows.append({"job": str(j)[:8], "name": nm[:80], "error": bool(getattr(r, "error", None))})
        st.dataframe(rows, use_container_width=True)

        failed = [(j, r) for j, r in items if getattr(r, "error", None)]
        if failed:
            st.error(f"{len(failed)} job(s) failed. Expand to inspect.")
            for j, r in failed[:5]:
                with st.expander(f"❌ {getattr(r,'name',str(j))} — job {str(j)}"):
                    err = getattr(r, "error", None)
                    if err is not None:
                        st.exception(err)
                    st.caption("Output (sanitized)")
                    st.json(_to_plain(getattr(r, "output", None)))
                    st.caption("Metadata (sanitized)")
                    st.json(_to_plain(getattr(r, "metadata", None)))
            st.stop()

        status.update(label="Extracting via Jobflow store…", state="running")

        # 1) Use the same pattern you posted: SETTINGS.JOB_STORE + connect() + query_one
        store = SETTINGS.JOB_STORE
        if store is None:
            raise RuntimeError("SETTINGS.JOB_STORE is None after run_locally; cannot query results.")

        # Some stores require explicit connect()
        try:
            store.connect()
        except Exception:
            pass

        # Query the most recent fit_elastic_tensor doc and load fields deeply
        doc = store.query_one(
            {"name": "fit_elastic_tensor"},
            properties=["output.elastic_tensor", "output.derived_properties"],
            load=True,
            sort={"completed_at": -1},
        )
        if doc is None:
            # fall back: deep scan rs to find a 6×6 tensor
            whole_plain = _to_plain(rs)
            Ccand = _find_tensor_anywhere(whole_plain)
            if Ccand is None:
                raise RuntimeError("No 'fit_elastic_tensor' document found in store and no tensor in responses.")
            Pcand = _props_from_any(whole_plain)
            C = _to_gpa(Ccand)
            P = Pcand or {}
        else:
            out = doc.get("output", {}) if isinstance(doc, dict) else {}
            et = (out.get("elastic_tensor", {}) if isinstance(out, dict) else {}) or {}
            # accept any reasonable field name for the 6×6
            Craw = None
            for key in ("ieee_format", "voigt", "matrix", "C_ij", "C"):
                if key in et and _looks_like_6x6(et[key]):
                    Craw = et[key]; break
            if Craw is None:
                raise RuntimeError("Store doc found but 'elastic_tensor' lacks a 6×6 matrix.")
            C = _to_gpa(Craw)
            P = out.get("derived_properties", {}) or {}

        # Backfill E, ν if missing
        K = P.get("k_vrh", None)
        G = P.get("g_vrh", None)
        if isinstance(K, (int,float)) and isinstance(G, (int,float)):
            denom = 3.0*K + G
            if denom != 0.0:
                P.setdefault("y_mod", 9.0*K*G/denom)
                P.setdefault("homogeneous_poisson", (3.0*K - 2.0*G) / (2.0*denom))

        payload = {"C_GPa": np.array(C, dtype=float).tolist(), "props": P}
        st.session_state["elastic_payload"] = payload

        status.update(label="Done ✅", state="complete")
        _render_results(payload)

    except Exception as e:
        st.error("Elastic workflow failed: {}".format(e))
        with st.expander("Exception", expanded=True):
            st.exception(e)
    finally:
        st.session_state["elastic_running"] = False


# -------------------- tiny helpers used after run --------------------
def _render_results(payload):
    C = np.array(payload["C_GPa"], dtype=float)
    P = payload["props"]

    st.markdown("**Elastic tensor C (GPa, 6×6 Voigt/IEEE)**")
    st.dataframe([[f"{v:.6f}" for v in row] for row in C], use_container_width=True)

    cA, cB, cC = st.columns(3)
    cA.metric("K_VRH (GPa)", _fmt3(P.get("k_vrh")))
    cB.metric("G_VRH (GPa)", _fmt3(P.get("g_vrh")))
    cC.metric("E (GPa)",     _fmt3(P.get("y_mod")))

    cD, cE = st.columns(2)
    cD.metric("Poisson ν", _fmt3(P.get("homogeneous_poisson")))
    cE.metric("Anisotropy AU", _fmt3(P.get("universal_anisotropy")))

    st.download_button("⬇️ JSON", json.dumps(payload, indent=2).encode("utf-8"), "elastic_results.json")
    st.download_button("⬇️ CSV", _pack_csv(C, P), "elastic_results.csv")

def _fmt3(x):
    return "{:.3f}".format(x) if isinstance(x, (int,float)) else "—"

def _find_tensor_anywhere(d):
    # quick deep-search for a 6×6 array in nested dict/list (used only as a fallback)
    def walk(x):
        if _looks_like_6x6(x):
            return x
        if isinstance(x, dict):
            for k in ("elastic_tensor","tensor","C_ij","C","voigt","ieee_format"):
                if k in x and _looks_like_6x6(x[k]):
                    return x[k]
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

def _props_from_any(d):
    if isinstance(d, dict):
        for key in ("derived_properties","properties"):
            if key in d and isinstance(d[key], dict):
                return d[key]
        for sub in ("output","data","result","results","elastic_data","elastic","metadata"):
            if sub in d and isinstance(d[sub], dict):
                got = _props_from_any(d[sub])
                if got:
                    return got
    return {}
