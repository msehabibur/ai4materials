# tabs/phonon_tab.py
from __future__ import annotations
import io, csv
import streamlit as st
from typing import Any, Dict
from pymatgen.core import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from atomate2.forcefields.flows.phonons import PhononMaker
from jobflow import run_locally, SETTINGS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PH_JSON = "phonon_json_bytes"
_PH_DOS_PNG = "phonon_dos_png"
_PH_BS_PNG  = "phonon_bs_png"
_PH_DOS_CSV = "phonon_dos_csv"

def _png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf.read()

def _dos_csv_bytes(ph_dos: PhononDos) -> bytes:
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["Frequency (THz)", "Total DOS"])
    freqs = ph_dos.frequencies
    total = ph_dos.densities
    for f, d in zip(freqs, total):
        w.writerow([f, d])
    return buf.getvalue().encode()

def phonon_tab(pmg_obj: Structure | None):
    st.subheader("🎵 Phonons — default force-field flow (MACE)")
    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer to compute phonons."); return

    c1, c2 = st.columns(2)
    with c1:
        min_length = st.number_input("Supercell min. length (Å)", 8.0, 40.0, 15.0, 1.0, key="ph_min_len")
    with c2:
        store_fc = st.checkbox("Store force constants", value=False, key="ph_store_fc")

    run_btn = st.button("Run Phonon", type="primary", key="ph_run_btn")

    # Show persisted outputs if any
    if st.session_state.get(_PH_DOS_PNG):
        st.image(st.session_state[_PH_DOS_PNG], caption="Phonon DOS (PNG)", use_column_width=True)
    if st.session_state.get(_PH_BS_PNG):
        st.image(st.session_state[_PH_BS_PNG], caption="Phonon Band Structure (PNG)", use_column_width=True)
    if st.session_state.get(_PH_DOS_CSV):
        st.download_button("⬇️ DOS (CSV)",
                           st.session_state[_PH_DOS_CSV], "phonon_dos.csv", key="ph_dl_dos_csv_persist")

    if not run_btn: return

    try:
        st.info("Generating phonon workflow…")
        flow = PhononMaker(
            min_length=float(min_length),
            store_force_constants=bool(store_fc)
        ).make(structure=pmg_obj)  # default engine under-the-hood uses MACE relaxers in this app

        st.info("Running locally… this may take a while for large cells.")
        run_locally(flow, create_folders=True)

        store = SETTINGS.JOB_STORE
        store.connect()

        result = store.query_one(
            {"name": "generate_frequencies_eigenvectors"},
            properties=["output.phonon_dos", "output.phonon_bandstructure"],
            load=True,
            sort={"completed_at": -1}
        )

        ph_bs  = PhononBandStructureSymmLine.from_dict(result["output"]["phonon_bandstructure"])
        ph_dos = PhononDos.from_dict(result["output"]["phonon_dos"])

        # Plot DOS
        dos_plot = PhononDosPlotter()
        dos_plot.add_dos("Phonon DOS", ph_dos)
        ax_dos = dos_plot.get_plot(); fig_dos = ax_dos.get_figure()
        dos_png = _png_bytes(fig_dos)

        # Plot Band Structure
        bs_plot = PhononBSPlotter(ph_bs)
        ax_bs   = bs_plot.get_plot(); fig_bs = ax_bs.get_figure()
        bs_png  = _png_bytes(fig_bs)

        # CSV for DOS
        dos_csv = _dos_csv_bytes(ph_dos)

        st.session_state[_PH_DOS_PNG] = dos_png
        st.session_state[_PH_BS_PNG]  = bs_png
        st.session_state[_PH_DOS_CSV] = dos_csv

        st.success("Phonon calculation finished ✅")
        st.image(dos_png, caption="Phonon DOS (PNG)", use_column_width=True)
        st.image(bs_png,  caption="Phonon Band Structure (PNG)", use_column_width=True)
        st.download_button("⬇️ DOS (CSV)", dos_csv, "phonon_dos.csv", key="ph_dl_dos_csv")

    except Exception as e:
        st.error(f"Phonon workflow failed: {e}")

__all__ = ["phonon_tab"]
