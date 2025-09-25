from __future__ import annotations
import json, gc, traceback, time
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

def _pretty_tensor(C):
    # C is expected in GPa if coming from Atomate2; we format nicely with units.
    if C is None:
        return "No tensor."
    arr = np.array(C, dtype=float)
    lines = []
    for row in arr:
        lines.append("  " + "  ".join(f"{v:10.3f}" for v in row))
    return "Elastic tensor C_ij (GPa):\n" + "\n".join(lines)

def _pretty_moduli(props):
    # props dictionary keys vary; we map common names and show units.
    if not props:
        return "No derived properties."
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
    lines = []
    warn_negative = False
    for key, label in fields:
        val = props.get(key, None)
        if val is None:
            continue
        try:
            v = float(val)
            if ("(GPa)" in label) and v < 0:
                warn_negative = True
            lines.append(f"{label}: {v:,.3f}")
        except Exception:
            lines.append(f"{label}: {val}")
    if warn_negative:
        lines.append("")
        lines.append("⚠️ One or more moduli are negative. This typically indicates an unstable/unstressed structure,")
        lines.append("    insufficient relaxation, or issues with the deformation data. Consider re-optimizing first,")
        lines.append("    using a better initial cell, or verifying symmetry/deformations.")
    return "\n".join(lines)

def elastic_tab(pmg_obj):
    st.subheader("Elastic Properties — CHGNet")
    CHGNetRelaxMaker, ElasticMaker, run_locally, SETTINGS, err = _lazy_import()
    if err:
        st.error("Dependencies missing. Ensure:")
        st.code("atomate2[phonons,forcefields]==0.0.18\njobflow==0.1.16\nphonopy==2.22.3\nspglib==2.5.0\nchgnet==0.4.0")
        st.code(str(err))
        return

    col1, col2 = st.columns(2)
    with col1:
        fmax = st.number_input("Relax fmax (eV/Å)", value=1e-4, min_value=1e-6, max_value=1e-2, step=1e-4, format="%.6f")
    with col2:
        run = st.button("Run elastic workflow", type="primary", disabled=(pmg_obj is None))
    if not run:
        return

    progress = st.progress(0, text="Preparing flow…")
    try:
        flow = ElasticMaker(
            bulk_relax_maker=CHGNetRelaxMaker(relax_cell=True,  relax_kwargs={"fmax": float(fmax)}),
            elastic_relax_maker=CHGNetRelaxMaker(relax_cell=False, relax_kwargs={"fmax": float(fmax)}),
        ).make(structure=pmg_obj)

        progress.progress(25, text="Running deformations…")
        _ = run_locally(flow, create_folders=True)

        progress.progress(85, text="Fitting elastic tensor…")
        time.sleep(0.2)

        store = SETTINGS.JOB_STORE
        store.connect()
        result = store.query_one(
            {"name": "fit_elastic_tensor"},
            properties=["output.elastic_tensor", "output.derived_properties"],
            load=True,
            sort={"completed_at": -1},
        )

        if not result:
            progress.progress(100, text="Done (no results)")
            st.warning("No results found.")
            return

        # Atomate2 returns elastic tensor and derived props typically in GPa.
        et = result["output"]["elastic_tensor"]
        dp = result["output"]["derived_properties"]

        progress.progress(100, text="Done")
        st.success("Results")

        # Show tensor (GPa) nicely
        tensor_to_show = et.get("ieee_format", et)
        st.code(_pretty_tensor(tensor_to_show))

        # Show derived properties with units
        st.code(_pretty_moduli(dp))

        del result, et, dp
        gc.collect()

    except Exception as exc:
        st.error("Elastic workflow failed.")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
