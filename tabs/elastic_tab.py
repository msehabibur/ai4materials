from __future__ import annotations
import json, gc, traceback
import streamlit as st

def _lazy_import():
    try:
        # Use the concrete CHGNet maker (works in Atomate2 forcefields)
        from atomate2.forcefields.jobs import CHGNetRelaxMaker
        from atomate2.forcefields.flows.elastic import ElasticMaker
        from jobflow import run_locally, SETTINGS
        return CHGNetRelaxMaker, ElasticMaker, run_locally, SETTINGS, None
    except Exception as exc:
        return None, None, None, None, exc

def elastic_tab(pmg_obj):
    st.subheader("Elastic Constants & Mechanical Stability (Atomate2)")

    CHGNetRelaxMaker, ElasticMaker, run_locally, SETTINGS, err = _lazy_import()
    if err:
        st.error("Atomate2/Jobflow not available. Ensure requirements include:")
        st.code("atomate2[phonons,forcefields]==0.0.18\njobflow==0.1.16\nchgnet==0.4.0")
        st.code(str(err))
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Backend fixed to CHGNet (stable).")
    with col2:
        fmax = st.number_input("Relax fmax (eV/Å)", value=1e-4, min_value=1e-6, max_value=1e-2, step=1e-4, format="%.6f")
    with col3:
        run = st.button("Run Elastic Workflow", type="primary", disabled=(pmg_obj is None))

    if not run:
        return

    try:
        flow = ElasticMaker(
            bulk_relax_maker=CHGNetRelaxMaker(relax_cell=True,  relax_kwargs={"fmax": float(fmax)}),
            elastic_relax_maker=CHGNetRelaxMaker(relax_cell=False, relax_kwargs={"fmax": float(fmax)}),
        ).make(structure=pmg_obj)

        st.info("Running elastic workflow locally…")
        _ = run_locally(flow, create_folders=True, raise_immediately=True)

        store = SETTINGS.JOB_STORE
        store.connect()
        result = store.query_one(
            {"name": "fit_elastic_tensor"},
            properties=["output.elastic_tensor", "output.derived_properties"],
            load=True,
            sort={"completed_at": -1},
        )

        if not result:
            st.warning("No results found yet.")
            return

        et = result["output"]["elastic_tensor"]
        dp = result["output"]["derived_properties"]
        st.success("Results")
        st.code(json.dumps(et.get("ieee_format", et), indent=2))
        st.code(json.dumps(dp, indent=2))

        del result, et, dp
        gc.collect()

    except Exception as exc:
        st.error("Elastic workflow failed.")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
