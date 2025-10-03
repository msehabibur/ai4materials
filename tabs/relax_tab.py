# tabs/relax_tab.py
from __future__ import annotations
import io, traceback
import numpy as np
import streamlit as st
from ase.optimize import BFGS, FIRE
from ase.constraints import UnitCellFilter
from pymatgen.io.ase import AseAtomsAdaptor

from core.model import get_calculator
from core.struct import atoms_from

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

_RELAX_PLOT_BYTES = "relax_plot_png"
_RELAX_LAST_CIF   = "relax_last_cif"

def _plot_E_vs_step(energies) -> bytes:
    plt.rcParams.update({"font.size": 16})
    fig = plt.figure(figsize=(5.6, 3.8))  # compact fixed space
    ax = plt.gca()
    sf = ScalarFormatter(useMathText=False); sf.set_scientific(False); sf.set_useOffset(False)
    ax.yaxis.set_major_formatter(sf)
    ax.plot(np.arange(len(energies)), energies, linewidth=2)
    ax.set_xlabel("Step"); ax.set_ylabel("Energy (eV)")
    ax.set_title("Optimization energy vs. step")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=170, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf.read()

def relax_tab(pmg_obj):
    st.subheader("Structure Optimization — MACE")
    if pmg_obj is None:
        st.info("Upload/select a structure in the viewer to optimize.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        optimizer_name = st.selectbox("Optimizer", ["FIRE", "BFGS"], key="relax_opt")
    with c2:
        fmax = st.number_input("Force tolerance fmax (eV/Å)", value=5e-5, min_value=1e-6, max_value=1e-2, step=1e-5, format="%.0e", key="relax_fmax")
    with c3:
        cell_mode = st.selectbox("Cell mode", ["Free volume (relax cell)", "Fixed volume (no cell relax)"], key="relax_cell")

    run_btn = st.button("Run Optimization", type="primary", key="relax_run")
    if not run_btn:
        # show persisted plot/downloads if exist
        if st.session_state.get(_RELAX_PLOT_BYTES):
            st.image(st.session_state[_RELAX_PLOT_BYTES], caption="Optimization energy vs. step")
        if st.session_state.get(_RELAX_LAST_CIF):
            st.download_button("Download optimized CIF", st.session_state[_RELAX_LAST_CIF], file_name="optimized.cif", key="relax_dl_cif_persist")
        return

    try:
        atoms = atoms_from(pmg_obj)
        atoms.calc = get_calculator("MACE", "default")

        if cell_mode.startswith("Free"):
            mobile = UnitCellFilter(atoms, hydrostatic_strain=True)
        else:
            mobile = atoms

        energies = []

        def log_step():
            try:
                energies.append(float(mobile.get_potential_energy()))
                pct = int(min(99, max(1, (len(energies) / 200) * 100)))  # not exact, but responsive
                progress.progress(pct, text=f"Optimizing… {pct}%")
                pct_label.write(f"**Progress:** {pct}%")
            except Exception:
                pass

        progress = st.progress(0, text="Starting…")
        pct_label = st.empty()

        if optimizer_name == "FIRE":
            opt = FIRE(mobile, logfile=None)
        else:
            opt = BFGS(mobile, logfile=None)

        opt.attach(log_step, interval=1)
        opt.run(fmax=float(fmax), steps=500)

        progress.progress(100, text="Done")
        pct_label.write("**Progress:** 100%")

        # Persist plot + CIF
        st.session_state[_RELAX_PLOT_BYTES] = _plot_E_vs_step(energies)
        st.image(st.session_state[_RELAX_PLOT_BYTES], caption="Optimization energy vs. step")

        opt_struct = AseAtomsAdaptor.get_structure(atoms)
        st.session_state[_RELAX_LAST_CIF] = opt_struct.to(fmt="cif").encode()
        st.download_button("Download optimized CIF", st.session_state[_RELAX_LAST_CIF], file_name="optimized.cif", key="relax_dl_cif")

        # Lattice summary
        a,b,c = opt_struct.lattice.abc
        al,be,ga = opt_struct.lattice.angles
        st.success(f"Optimized lattice: a={a:.4f}, b={b:.4f}, c={c:.4f} Å | α={al:.2f}, β={be:.2f}, γ={ga:.2f}°")

    except Exception as exc:
        st.error(f"Optimization failed: {exc}")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
