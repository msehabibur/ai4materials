# tabs/elastic_tab.py
from __future__ import annotations

import os, json, io, numbers
import numpy as np
import streamlit as st
from monty.json import jsanitize
from pymatgen.core import Structure

from jobflow import run_locally, SETTINGS
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import MACERelaxMaker

# ---------- helpers ----------
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
    arr = np.array(x, dtype=float)
    # Heuristic: values much larger than ~1e5 are probably in Pa → convert to GPa
    try:
        if np.nanmax(np.abs(arr)) > 1e5:
            arr *= 1e-9
    except Exception:
        pass
    return arr

def _props_to_gpa(props: dict) -> dict:
    """Convert common elastic properties from Pa to GPa if needed."""
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

def _pack_csv(C_gpa: np.ndarray, props: dict) -> bytes:
    rows = []
    rows.append(["Elastic tensor C (GPa) — 6×6 (Voigt/IEEE)"])
    for r in C_gpa:
        rows.append([f"{v:.6f}" for v in r])
    rows.append([])
    rows.append(["Property", "Value (GPa or dimensionless)"])
    keys = [
        "k_voigt","k_reuss","k_vrh",
        "g_voigt","g_reuss","g_vrh",
        "y_mod","homogeneous_poisson","universal_anisotropy",
    ]
    for k in keys:
        v = props.get(k, None)
        if isinstance(v, (int, float)):
            rows.append([k, f"{v:.6f}"])
        else:
            rows.append([k, "—"])
    import csv
    s = io.StringIO()
    w = csv.writer(s)
    w.writerows(rows)
    return s.getvalue().encode("utf-8")

def _render_results(C_gpa: np.ndarray, P: dict):
    # Results card
    st.markdown("### Elastic tensor (GPa)")
    st.dataframe([[f"{v:.6f}" for v in row] for row in C_gpa], use_container_width=True)

    # Heatmap (nice, compact)
    try:
        import plotly.graph_objects as go
        fig = go.Figure(
            data=go.Heatmap(
                z=C_gpa,
                x=["C11","C12","C13","C14","C15","C16"],
                y=["C11","C12","C13","C14","C15","C16"],
                zmid=float(np.nanmean(C_gpa)),
                colorbar_title="GPa",
            )
        )
        fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=280)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        pass

    # Summary metrics
    cA, cB, cC = st.columns(3)
    cA.metric("K_VRH (GPa)", _fmt3(P.get("k_vrh")))
    cB.metric("G_VRH (GPa)", _fmt3(P.get("g_vrh")))
    cC.metric("E (GPa)",     _fmt3(P.get("y_mod")))
    cD, cE = st.columns(2)
    cD.metric("Poisson ν", _fmt3(P.get("homogeneous_poisson")))
    cE.metric("Anisotropy AU", _fmt3(P.get("universal_anisotropy")))

    # Downloads
    st.download_button("⬇️ JSON", json.dumps({"C_GPa": C_gpa.tolist(), "props": P}, indent=2).encode("utf-8"),
                       "elastic_results.json")
    st.download_button("⬇️ CSV", _pack_csv(C_gpa, P), "elastic_results.csv")

def _fmt3(x):
    return "{:.3f}".format(x) if isinstance(x, (int, float)) else "—"

# ---------- main tab ----------
def elastic_tab(pmg_obj: Structure | None):
    st.subheader("🧱 Elastic — MACE")
    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer first.")
        return

    st.session_state.setdefault("elastic_running", False)
    st.session_state.setdefault("elastic_payload", None)
    show_debug = st.toggle("Show debug details", value=False)

    disabled = st.session_state["elastic_running"]
    c1, c2 = st.columns(2)
    with c1:
        fmax = st.number_input(
            "Relax fmax (eV/Å)", min_value=1e-8, max_value=1e-1,
            value=1e-5, step=1e-8, format="%.0e", disabled=disabled
        )
    with c2:
        allow_cell = st.selectbox(
            "Relax cell?", ["Yes (recommended)", "No (fixed cell)"],
            index=0, disabled=disabled
        ).startswith("Yes")

    run_btn = st.button("Run Elastic Workflow", type="primary", disabled=disabled)

    # Show previous results (quiet)
    if (not st.session_state["elastic_running"]) and st.session_state["elastic_payload"]:
        C_gpa = np.array(st.session_state["elastic_payload"]["C_GPa"], dtype=float)
        P     = st.session_state["elastic_payload"]["props"]
        _render_results(C_gpa, P)

    if (not run_btn) or st.session_state["elastic_running"]:
        return

    # Run
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

        # IMPORTANT: follow your working pattern — do NOT pass store=
        rs = run_locally(flow, create_folders=True, ensure_success=False)

        # Show failing jobs only when debug is on
        items = list(rs.items()) if isinstance(rs, dict) else list(enumerate(rs))
        failed = [(j, r) for j, r in items if getattr(r, "error", None)]
        if show_debug:
            rows = []
            for j, r in items:
                nm = getattr(r, "name", None) or str(j)
                rows.append({"job": str(j)[:8], "name": nm[:80], "error": bool(getattr(r, "error", None))})
            st.dataframe(rows, use_container_width=True)

        if failed:
            st.error(f"{len(failed)} job(s) failed.")
            if show_debug:
                for j, r in failed[:5]:
                    with st.expander(f"❌ {getattr(r,'name',str(j))} — job {str(j)}"):
                        if getattr(r, "error", None):
                            st.exception(getattr(r, "error"))
                        st.caption("Output (sanitized)")
                        st.json(_to_plain(getattr(r, "output", None)))
                        st.caption("Metadata (sanitized)")
                        st.json(_to_plain(getattr(r, "metadata", None)))
            st.stop()

        # Extract results using the Jobflow store like in your working snippet
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
            # fallback: deep-scan responses
            if show_debug:
                st.warning("Store query returned None; scanning run responses as fallback.")
            plain = _to_plain(rs)
            C_raw = None
            # walk responses to find a 6x6 matrix
            def walk(x):
                nonlocal C_raw
                if C_raw is not None:
                    return
                if _looks_like_6x6(x):
                    C_raw = x; return
                if isinstance(x, dict):
                    # try common keys first
                    for k in ("elastic_tensor","tensor","C_ij","C","voigt","ieee_format","output","data","results","result","metadata"):
                        if k in x:
                            walk(x[k])
                    if C_raw is None:
                        for v in x.values():
                            walk(v)
                elif isinstance(x, (list, tuple)):
                    for v in x:
                        walk(v)
            walk(plain)
            if C_raw is None:
                raise RuntimeError("No 'fit_elastic_tensor' document and no 6×6 tensor found in responses.")
            P = {}
        else:
            out = doc.get("output", {}) if isinstance(doc, dict) else {}
            et  = out.get("elastic_tensor", {}) if isinstance(out, dict) else {}
            C_raw = _pick_c_6x6(et)
            if C_raw is None:
                raise RuntimeError("Elastic tensor found, but no 6×6 field among ieee_format/voigt/matrix/C_ij/C.")
            P = out.get("derived_properties", {}) or {}

        # Convert everything to GPa and compute E, ν consistently
        C_gpa = _as_gpa_array(C_raw)
        P     = _props_to_gpa(P)

        K = P.get("k_vrh"); G = P.get("g_vrh")
        if isinstance(K, (int, float)) and isinstance(G, (int, float)):
            denom = 3.0*K + G
            if denom != 0.0:
                P.setdefault("y_mod", 9.0*K*G/denom)  # E in GPa
                P.setdefault("homogeneous_poisson", (3.0*K - 2.0*G)/(2.0*denom))

        st.session_state["elastic_payload"] = {"C_GPa": C_gpa.tolist(), "props": P}

        # Render clean UI
        _render_results(C_gpa, P)

        if show_debug:
            st.caption("Done")

    except Exception as e:
        st.error(f"Elastic workflow failed: {e}")
        with st.expander("Exception", expanded=True):
            st.exception(e)
    finally:
        st.session_state["elastic_running"] = False
