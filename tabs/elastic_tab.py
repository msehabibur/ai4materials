# tabs/elastic_tab.py
from __future__ import annotations
import gc, traceback, time
import numpy as np
import streamlit as st

def _lazy_import():
    try:
        from atomate2.forcefields.jobs import CHGNetRelaxMaker
        from atomate2.forcefields.flows.elastic import ElasticMaker
        from jobflow import run_locally, SETTINGS
        return CHGNetRelaxMaker, ElasticMaker, run_locally, SETTINGS, None
    except Exception as exc:
        return None, None, None, None, exc

def _guess_scale_to_gpa(C):
    arr = np.abs(np.array(C, dtype=float))
    med = np.median(np.diag(arr)) if arr.shape[0] >= 3 else np.median(arr)
    if med > 1e9:   return 1/1e9, "Pa"
    if med > 1e4:   return 1/1e3, "MPa"
    return 1.0, "GPa"

def _pretty_tensor_GPa(C):
    arr = np.array(C, dtype=float)
    s, units = _guess_scale_to_gpa(arr)
    arr = arr * s
    lines = ["Elastic tensor C_ij (GPa):"]
    for row in arr:
        lines.append("  " + "  ".join(f"{v:10.3f}" for v in row))
    return "\n".join(lines), units

def _scale_props_to_gpa(props):
    if not props: return props, "GPa"
    candidates = [props.get("k_vrh"), props.get("g_vrh"), props.get("y_mod")]
    rep = [float(x) for x in candidates if isinstance(x, (int,float))]
    units = "GPa"; s = 1.0
    if rep:
        med = np.median(np.abs(rep))
        if med > 1e9: s, units = 1/1e9, "Pa"
        elif med > 1e4: s, units = 1/1e3, "MPa"
    out = {}
    for k,v in props.items():
        if v is None: out[k] = None; continue
        try:
            out[k] = float(v) * s if k != "homogeneous_poisson" and "anisotropy" not in k else float(v)
        except Exception:
            out[k] = v
    return out, units

def _pretty_moduli(props_gpa, units_detected):
    fields = [
        ("k_voigt", "Bulk K_V (GPa)"),
        ("k_reuss", "Bulk K_R (GPa)"),
        ("k_vrh", "Bulk K_VRH (GPa)"),
        ("g_voigt", "Shear G_V (GPa)"),
        ("g_reuss", "Shear G_R (GPa)"),
        ("g_vrh", "Shear G_VRH (GPa)"),
        ("y_mod", "Young's modulus (GPa)"),
        ("homogeneous_poisson", "Poisson ratio (–)"),
        ("universal_anisotropy", "Universal anisotropy (–)"),
    ]
    lines = [f"Detected units: {units_detected} → shown in GPa"]
    warn_negative = False
    for key, label in fields:
        val = props_gpa.get(key, None)
        if val is None: continue
        try:
            v = float(val)
            if "(GPa)" in label and v < 0: warn_negative = True
            lines.append(f"{label}: {v:,.3f}")
        except Exception:
            lines.append(f"{label}: {val}")
    if warn_negative:
        lines += [
            "",
            "⚠️ One or more moduli are negative. This usually means the structure is mechanically unstable,",
            "   insufficiently relaxed, or the deformation fit failed. Re-optimize and retry."
        ]
    return "\n".join(lines)

def elastic_tab(pmg_obj, model_family: str, low_mem: bool):
    st.subheader("Elastic Properties")
    CHGNetRelaxMaker, ElasticMaker, run_locally, SETTINGS, err = _lazy_import()
    if err:
        st.error("Dependencies missing. Ensure: atomate2[phonons,forcefields], jobflow, phonopy, spglib.")
        st.code(str(err)); return

    if model_family != "CHGNet":
        st.info("Elastic flow uses Atomate2 force-field workflows which currently support CHGNet. "
                "Switch family to CHGNet in the sidebar to enable this tab.")
        return

    # Slightly looser default fmax for per-deformation relax in low-mem mode
    default_fmax = 2e-4 if low_mem else 1e-4

    col1, col2 = st.columns(2)
    with col1:
        fmax = st.number_input("Relax fmax (eV/Å)", value=float(default_fmax), min_value=1e-6, max_value=1e-2,
                               step=1e-4, format="%.6f", key="el_fmax")
    with col2:
        run = st.button("Run elastic workflow", type="primary", disabled=(pmg_obj is None), key="el_run")
    if not run: return

    progress = st.progress(0, text="Preparing flow…")
    pct_label = st.empty()

    try:
        progress.progress(10, text="Preparing flow…"); pct_label.write("**Progress:** 10%")
        flow = ElasticMaker(
            bulk_relax_maker=CHGNetRelaxMaker(relax_cell=True,  relax_kwargs={"fmax": float(fmax)}),
            elastic_relax_maker=CHGNetRelaxMaker(relax_cell=False, relax_kwargs={"fmax": float(fmax)}),
        ).make(structure=pmg_obj)

        progress.progress(30, text="Running deformations…"); pct_label.write("**Progress:** 30%")
        _ = run_locally(flow, create_folders=True)

        progress.progress(85, text="Fitting elastic tensor…"); pct_label.write("**Progress:** 85%")
        time.sleep(0.2)

        store = SETTINGS.JOB_STORE
        store.connect()
        result = store.query_one(
            {"name": "fit_elastic_tensor"},
            properties=["output.elastic_tensor", "output.derived_properties"],
            load=True,
            sort={"completed_at": -1},
        )

        progress.progress(95, text="Formatting results…"); pct_label.write("**Progress:** 95%")
        if not result:
            progress.progress(100, text="Done (no results)"); pct_label.write("**Progress:** 100%")
            st.warning("No results found."); return

        et = result["output"]["elastic_tensor"]
        dp = result["output"]["derived_properties"]

        tensor_to_show = et.get("ieee_format", et)
        tensor_text, units_t = _pretty_tensor_GPa(tensor_to_show)
        props_scaled, units_p = _scale_props_to_gpa(dp)

        st.code(tensor_text)
        st.code(_pretty_moduli(props_scaled, units_p))

        del result, et, dp
        gc.collect()

        progress.progress(100, text="Done"); pct_label.write("**Progress:** 100%")

    except Exception as exc:
        st.error("Elastic workflow failed.")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
