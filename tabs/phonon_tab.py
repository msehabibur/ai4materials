from __future__ import annotations
import io, csv, gc, traceback, time
import streamlit as st

def _lazy_import():
    try:
        from atomate2.forcefields.flows.phonons import PhononMaker
        from jobflow import run_locally, SETTINGS
        return PhononMaker, run_locally, SETTINGS, None
    except Exception as exc:
        return None, None, None, exc

def phonon_tab(pmg_obj):
    st.subheader("Phonons — CHGNet")
    PhononMaker, run_locally, SETTINGS, err = _lazy_import()
    if err:
        st.error("Dependencies missing. Ensure:")
        st.code("atomate2[phonons,forcefields]==0.0.18\njobflow==0.1.16\nphonopy==2.22.3\nspglib==2.5.0")
        st.code(str(err))
        return

    col1, col2 = st.columns(2)
    with col1:
        min_len = st.number_input("Supercell min length (Å)", value=12.0, min_value=8.0, max_value=25.0, step=0.5)
    with col2:
        store_fc = st.checkbox("Store force constants (higher RAM)", value=False)

    run = st.button("Run phonon workflow", type="primary", disabled=(pmg_obj is None))
    if not run:
        return

    progress = st.progress(0, text="Preparing flow…")
    try:
        flow = PhononMaker(min_length=float(min_len), store_force_constants=bool(store_fc)).make(structure=pmg_obj)
        progress.progress(20, text="Running jobs…")
        _ = run_locally(flow, create_folders=True)

        progress.progress(80, text="Collecting results…")
        store = SETTINGS.JOB_STORE
        store.connect()
        result = store.query_one(
            {"name": "generate_frequencies_eigenvectors"},
            properties=["output.phonon_dos", "output.phonon_bandstructure"],
            load=True,
            sort={"completed_at": -1},
        )

        if not result:
            progress.progress(100, text="Done (no results)")
            st.warning("No phonon results found.")
            return

        from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
        from pymatgen.phonon.dos import PhononDos
        from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
        import matplotlib
        matplotlib.use("Agg")

        ph_bs = PhononBandStructureSymmLine.from_dict(result['output']['phonon_bandstructure'])
        ph_dos = PhononDos.from_dict(result['output']['phonon_dos'])
        del result; gc.collect()

        # DOS
        dos_plot = PhononDosPlotter()
        dos_plot.add_dos("Phonon DOS", ph_dos)
        ax_dos = dos_plot.get_plot()
        fig_dos = ax_dos.get_figure()
        buf_dos = io.BytesIO()
        fig_dos.savefig(buf_dos, dpi=300, bbox_inches="tight")
        buf_dos.seek(0)
        st.image(buf_dos.read(), caption="Phonon DOS", use_container_width=True)

        # Band structure
        bs_plot = PhononBSPlotter(ph_bs)
        ax_bs = bs_plot.get_plot()
        fig_bs = ax_bs.get_figure()
        buf_bs = io.BytesIO()
        fig_bs.savefig(buf_bs, dpi=300, bbox_inches="tight")
        buf_bs.seek(0)
        st.image(buf_bs.read(), caption="Phonon Band Structure", use_container_width=True)

        # CSV exports
        dos_csv = io.StringIO()
        w = csv.writer(dos_csv)
        w.writerow(["Frequency (THz)", "Total DOS"])
        for fval, dval in zip(ph_dos.frequencies, ph_dos.densities):
            w.writerow([fval, dval])
        st.download_button("⬇️ Download Phonon DOS (CSV)", dos_csv.getvalue().encode(), file_name="phonon_dos.csv")

        bands = ph_bs.bands
        distances = ph_bs.distance
        bands_T = bands.T
        bs_csv = io.StringIO()
        w = csv.writer(bs_csv)
        header = ["Distance (1/Å)"] + [f"Branch {i+1} (THz)" for i in range(bands.shape[0])]
        w.writerow(header)
        for i, d in enumerate(distances):
            row = [d] + list(bands_T[i])
            w.writerow(row)
        w.writerow([]); w.writerow(["High Symmetry Points"])
        for label, dist in ph_bs.labels_dict.items():
            w.writerow([label, dist])
        st.download_button("⬇️ Download Phonon Bands (CSV)", bs_csv.getvalue().encode(), file_name="phonon_bandstructure.csv")

        progress.progress(100, text="Done")

    except Exception as exc:
        st.error("Phonon workflow failed.")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
