from __future__ import annotations
import io, csv, gc, traceback
import streamlit as st

def _lazy_import():
    try:
        from atomate2.forcefields.flows.phonons import PhononMaker
        from jobflow import run_locally, SETTINGS
        return PhononMaker, run_locally, SETTINGS, None
    except Exception as exc:
        return None, None, None, exc

def phonon_tab(pmg_obj):
    st.subheader("Phonons (Atomate2): DOS & Band Structure")

    PhononMaker, run_locally, SETTINGS, err = _lazy_import()
    if err:
        st.error("Atomate2/Jobflow not available. Add to requirements: `atomate2`, `jobflow`")
        return

    col1, col2 = st.columns(2)
    with col1:
        min_len = st.number_input("Supercell min length (Å)", value=12.0, min_value=8.0, max_value=25.0, step=0.5)
    with col2:
        store_fc = st.checkbox("Store force constants (bigger RAM)", value=False)

    run = st.button("Run Phonon Workflow", type="primary", disabled=(pmg_obj is None))
    if not run:
        return

    try:
        flow = PhononMaker(
            min_length=float(min_len),
            store_force_constants=bool(store_fc),
        ).make(structure=pmg_obj)

        st.info("Running locally via jobflow…")
        _ = run_locally(flow, create_folders=True)

        # Query results
        store = SETTINGS.JOB_STORE
        store.connect()
        result = store.query_one(
            {"name": "generate_frequencies_eigenvectors"},
            properties=["output.phonon_dos", "output.phonon_bandstructure"],
            load=True,
            sort={"completed_at": -1},
        )

        if not result:
            st.warning("No phonon results found yet.")
            return

        from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
        from pymatgen.phonon.dos import PhononDos
        from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
        import matplotlib
        matplotlib.use("Agg")

        ph_bs = PhononBandStructureSymmLine.from_dict(result['output']['phonon_bandstructure'])
        ph_dos = PhononDos.from_dict(result['output']['phonon_dos'])

        # Free the huge dict early
        del result
        gc.collect()

        # DOS plot
        dos_plot = PhononDosPlotter()
        dos_plot.add_dos("Phonon DOS", ph_dos)
        ax_dos = dos_plot.get_plot()
        fig_dos = ax_dos.get_figure()
        buf_dos = io.BytesIO()
        fig_dos.savefig(buf_dos, dpi=300, bbox_inches="tight")
        buf_dos.seek(0)
        st.image(buf_dos.read(), caption="Phonon DOS", use_container_width=True)

        # Band structure plot
        bs_plot = PhononBSPlotter(ph_bs)
        ax_bs = bs_plot.get_plot()
        fig_bs = ax_bs.get_figure()
        buf_bs = io.BytesIO()
        fig_bs.savefig(buf_bs, dpi=300, bbox_inches="tight")
        buf_bs.seek(0)
        st.image(buf_bs.read(), caption="Phonon Band Structure", use_container_width=True)

        # CSV exports
        # DOS
        dos_csv = io.StringIO()
        writer = csv.writer(dos_csv)
        writer.writerow(["Frequency (THz)", "Total DOS"])
        for fval, dval in zip(ph_dos.frequencies, ph_dos.densities):
            writer.writerow([fval, dval])
        st.download_button("⬇️ Download Phonon DOS (CSV)", dos_csv.getvalue().encode(), file_name="phonon_dos.csv")

        # Bands
        bands = ph_bs.bands  # (n_branches, n_points)
        distances = ph_bs.distance
        bands_T = bands.T
        bs_csv = io.StringIO()
        writer = csv.writer(bs_csv)
        header = ["Distance (1/Å)"] + [f"Branch {i+1} (THz)" for i in range(bands.shape[0])]
        writer.writerow(header)
        for i, d in enumerate(distances):
            row = [d] + list(bands_T[i])
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["High Symmetry Points"])
        for label, dist in ph_bs.labels_dict.items():
            writer.writerow([label, dist])
        st.download_button("⬇️ Download Phonon Bands (CSV)", bs_csv.getvalue().encode(), file_name="phonon_bandstructure.csv")

        # Free plotting objects
        del ph_bs, ph_dos, bands, bands_T, distances, fig_bs, fig_dos
        gc.collect()

        st.success("Phonon analyses complete. Defaults tuned for low memory.")

    except Exception as exc:
        st.error("Phonon workflow failed.")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
