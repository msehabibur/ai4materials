# tabs/phonon_tab.py
from __future__ import annotations
import io, traceback
import streamlit as st

from pymatgen.core import Structure
from jobflow import run_locally, SETTINGS
from atomate2.forcefields.flows.phonons import PhononMaker
from atomate2.forcefields.jobs import MACERelaxMaker

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PH_PNG_BS  = "phonon_bs_png"
_PH_PNG_DOS = "phonon_dos_png"

def _render_fixed_png(fig, width=560, height=360, dpi=160) -> bytes:
    fig.set_size_inches(width/96, height/96)  # approx pixels to inches at 96dpi
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf.read()

def phonon_tab(pmg_obj: Structure | None):
    st.subheader("Phonons — MACE")
    if pmg_obj is None:
        st.info("Upload/select a structure first.")
        return

    c1, c2 = st.columns(2)
    with c1:
        min_len = st.number_input("Min. supercell length (Å)", value=15.0, min_value=5.0, max_value=200.0, step=1.0, key="ph_minlen")
    with c2:
        store_fc = st.checkbox("Store force constants (heavier)", value=False, key="ph_storefc")

    run_btn = st.button("Run Phonon Flow", type="primary", key="ph_run")

    # show persisted images if present
    if st.session_state.get(_PH_PNG_BS):
        st.image(st.session_state[_PH_PNG_BS], caption="Phonon band structure", width=560)
    if st.session_state.get(_PH_PNG_DOS):
        st.image(st.session_state[_PH_PNG_DOS], caption="Phonon DOS", width=560)

    if not run_btn:
        return

    try:
        maker = PhononMaker(
            min_length=float(min_len),
            store_force_constants=bool(store_fc)
        )
        # note: PhononMaker in atomate2 0.0.14 does NOT take relax_maker arg.
        flow = maker.make(structure=pmg_obj)
        progress = st.progress(0, text="Running phonon flow… 0%")

        responses = run_locally(flow, create_folders=True)
        progress.progress(70, text="Querying results… 70%")

        store = SETTINGS.JOB_STORE; store.connect()
        result = store.query_one(
            {"name": "generate_frequencies_eigenvectors"},
            properties=["output.phonon_dos", "output.phonon_bandstructure"],
            load=True, sort={"completed_at": -1}
        )
        from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
        from pymatgen.phonon.dos import PhononDos
        from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter

        ph_bs = PhononBandStructureSymmLine.from_dict(result['output']['phonon_bandstructure'])
        ph_dos = PhononDos.from_dict(result['output']['phonon_dos'])

        # Smaller legends and fixed image sizes
        bs_plot = PhononBSPlotter(ph_bs)
        ax_bs = bs_plot.get_plot()
        ax_bs.legend(fontsize=9, loc="best") if ax_bs.get_legend() else None
        st.session_state[_PH_PNG_BS] = _render_fixed_png(ax_bs.get_figure(), width=560, height=360, dpi=170)

        dos_plot = PhononDosPlotter()
        dos_plot.add_dos("Total", ph_dos)
        ax_dos = dos_plot.get_plot()
        leg = ax_dos.get_legend()
        if leg: leg.set_fontsize(9)
        st.session_state[_PH_PNG_DOS] = _render_fixed_png(ax_dos.get_figure(), width=560, height=360, dpi=170)

        progress.progress(100, text="Done")
        st.image(st.session_state[_PH_PNG_BS], caption="Phonon band structure", width=560)
        st.image(st.session_state[_PH_PNG_DOS], caption="Phonon DOS", width=560)

    except Exception as exc:
        st.error(f"Phonon flow failed: {exc}")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
