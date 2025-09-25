# tabs/phonon_tab.py
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

def phonon_tab(pmg_obj, model_family: str, low_mem: bool):
    st.subheader("Phonons")

    PhononMaker, run_locally, SETTINGS, err = _lazy_import()
    if err:
        st.error("Dependencies missing. Ensure: atomate2[phonons,forcefields], jobflow, phonopy, spglib.")
        st.code(str(err)); return

    if model_family != "CHGNet":
        st.info("Phonon flow uses Atomate2 force-field workflows which currently support CHGNet. "
                "Switch family to CHGNet in the sidebar to enable this tab.")
        return

    # Low-memory defaults
    default_min_len = 10.0 if low_mem else 12.0
    default_store_fc = False  # storing force constants is memory heavy

    col1, col2 = st.columns(2)
    with col1:
        min_len = st.number_input("Supercell min length (Å)", value=default_min_len, min_value=8.0, max_value=25.0, step=0.5, key="ph_minlen")
    with col2:
        store_fc = st.checkbox("Store force constants (higher RAM)", value=default_store_fc, key="ph_storefc")

    run = st.button("Run phonon workflow", type="primary", disabled=(pmg_obj is None), key="ph_run")
    if not run: return

    progress = st.progress(0, text="Preparing flow…")
    pct_label = st.empty()

    try:
        progress.progress(10, text="Preparing flow…"); pct_label.write("**Progress:** 10%")
        flow = PhononMaker(min_length=float(min_len), store_force_constants=bool(store_fc)).make(structure=pmg_obj)

        progress.progress(25, text="Submitting jobs…"); pct_label.write("**Progress:** 25%")
        _ = run_locally(flow, create_folders=True)

        progress.progress(80, text="Collecting results…"); pct_label.write("**Progress:** 80%")
        store = SETTINGS.JOB_STORE
        store.connect()
        result = store.query_one(
            {"name": "generate_frequencies_eigenvectors"},
            properties=["output.phonon_dos", "output.phonon_bandstructure"],
            load=True,
            sort={"completed_at": -1},
        )

        progress.progress(90, text="Rendering plots…"); pct_label.write("**Progress:** 90%")
        if not result:
            progress.progress(100, text="Done (no results)"); pct_label.write("**Progress:** 100%")
            st.warning("No phonon results found."); return

        from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
        from pymatgen.phonon.dos import PhononDos
        from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ph_bs = PhononBandStructureSymmLine.from_dict(result['output']['phonon_bandstructure'])
        ph_dos = PhononDos.from_dict(result['output']['phonon_dos'])
        del result; gc.collect()

        # DOS
        plt.rcParams.update({"font.size": 16})
        dos_plot = PhononDosPlotter(); dos_plot.add_dos("Phonon DOS", ph_dos)
        ax_dos = dos_plot.get_plot()
        fig_dos = ax_dos.get_figure(); fig_dos.set_size_inches(5.0, 3.6)
        buf_dos = io.BytesIO(); fig_dos.savefig(buf_dos, dpi=160, bbox_inches="tight"); buf_dos.seek(0)
        st.image(buf_dos.read(), caption="Phonon DOS", use_container_width=False); plt.close(fig_dos)

        # Band structure
        bs_plot = PhononBSPlotter(ph_bs)
        ax_bs = bs_plot.get_plot()
        fig_bs = ax_bs.get_figure(); fig_bs.set_size_inches(5.0, 3.6)
        buf_bs = io.BytesIO(); fig_bs.savefig(buf_bs, dpi=160, bbox_inches="tight"); buf_bs.seek(0)
        st.image(buf_bs.read(), caption="Phonon Band Structure", use_container_width=False); plt.close(fig_bs)

        # CSV exports
        import csv
        dos_csv = io.StringIO(); w = csv.writer(dos_csv)
        w.writerow(["Frequency (THz)", "Total DOS"])
        for fval, dval in zip(ph_dos.frequencies, ph_dos.densities):
            w.writerow([fval, dval])
        st.download_button("⬇️ Phonon DOS (CSV)", dos_csv.getvalue().encode(), file_name="phonon_dos.csv", key="ph_dos_csv")

        bands = ph_bs.bands; distances = ph_bs.distance; bands_T = bands.T
        bs_csv = io.StringIO(); w = csv.writer(bs_csv)
        header = ["Distance (1/Å)"] + [f"Branch {i+1} (THz)" for i in range(bands.shape[0])]
        w.writerow(header)
        for i, d in enumerate(distances):
            row = [d] + list(bands_T[i]); w.writerow(row)
        w.writerow([]); w.writerow(["High Symmetry Points"])
        for label, dist in ph_bs.labels_dict.items():
            w.writerow([label, dist])
        st.download_button("⬇️ Phonon Bands (CSV)", bs_csv.getvalue().encode(), file_name="phonon_bandstructure.csv", key="ph_bs_csv")

        progress.progress(100, text="Done"); pct_label.write("**Progress:** 100%")

    except Exception as exc:
        st.error("Phonon workflow failed.")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
