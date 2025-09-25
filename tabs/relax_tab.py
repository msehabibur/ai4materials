# tabs/relax_tab.py
from __future__ import annotations
import io, traceback, numpy as np
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter  # << added for robust plain ticks

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

from ase.optimize import BFGS, LBFGS, FIRE
from ase.constraints import StrainFilter, UnitCellFilter
from ase.io import write as ase_write
from ase.optimize.optimize import Optimizer

from core.struct import atoms_from
from core.model import get_calculator

# session keys
_PLOT_KEY = "relax_plot_png"
_CIF_KEY = "relaxed_cif"
_XYZ_KEY = "relax_traj_xyz"

def _lattice_summary(s: Structure) -> str:
    a, b, c = s.lattice.abc
    α, β, γ = s.lattice.angles
    return f"a={a:.3f}, b={b:.3f}, c={c:.3f} Å | α={α:.2f}, β={β:.2f}, γ={γ:.2f}°"

def _plot_energy(energies: list[float]) -> bytes | None:
    if not energies:
        return None
    plt.rcParams.update({"font.size": 18})
    fig = plt.figure(figsize=(5.0, 3.6))
    ax = plt.gca()
    # Force fully plain numeric format (no scientific/engineering, no offset)
    sf = ScalarFormatter(useMathText=False)
    sf.set_scientific(False)
    sf.set_useOffset(False)
    ax.yaxis.set_major_formatter(sf)

    xs = np.arange(len(energies))
    ax.plot(xs, energies, marker="o", linewidth=2)
    ax.set_xlabel("Optimization step"); ax.set_ylabel("Energy (eV)"); ax.set_title("Energy vs step")
    buf = io.BytesIO(); fig.savefig(buf, dpi=160, bbox_inches="tight"); plt.close(fig)
    buf.seek(0); return buf.read()

def _make_optimizer(name: str, atoms_or_filter) -> Optimizer:
    if name == "BFGS": return BFGS(atoms_or_filter, logfile=None, maxstep=0.2)
    if name == "LBFGS": return LBFGS(atoms_or_filter, logfile=None)
    if name == "FIRE": return FIRE(atoms_or_filter, logfile=None)
    return BFGS(atoms_or_filter, logfile=None, maxstep=0.2)

def relax_tab(pmg_obj, model_family: str, variant: str, low_mem: bool):
    st.subheader("Structure Optimization")

    # show persisted artifacts first (so they don't disappear after downloads/reruns)
    if st.session_state.get(_PLOT_KEY):
        st.image(st.session_state[_PLOT_KEY], caption="Energy vs optimization step", use_container_width=False)
    if st.session_state.get(_CIF_KEY):
        st.download_button("⬇️ Download relaxed CIF", st.session_state[_CIF_KEY],
                           "relaxed_structure.cif", key="relax_dl_cif_persist")
    if st.session_state.get(_XYZ_KEY):
        st.download_button("⬇️ Download trajectory (XYZ)", st.session_state[_XYZ_KEY],
                           "relax_trajectory.xyz", key="relax_dl_xyz_persist")

    # Defaults adjusted by low-memory mode
    default_steps = 120 if low_mem else 200
    default_stride = 20 if low_mem else 10

    c1, c2, c3 = st.columns(3)
    with c1:
        # ↓ allow very small fmax (≥ 1e-6) and show with 6 decimals
        fmax = st.number_input(
            "Force convergence (eV/Å)",
            value=0.000050, min_value=0.000001, max_value=1.0,
            step=0.000001, format="%.6f", key="relax_fmax"
        )
    with c2:
        max_steps = st.number_input("Max steps", value=default_steps, min_value=10, max_value=5000,
                                    step=10, key="relax_maxsteps")
    with c3:
        optimizer = st.selectbox("Optimizer", ["BFGS", "LBFGS", "FIRE"], index=0, key="relax_opt")

    mode = st.selectbox(
        "Optimization mode",
        ["Positions only", "Variable cell (shape + volume)", "Cell shape only (approx. fixed volume)"],
        index=0, key="relax_mode")

    save_xyz = st.checkbox("Stream trajectory to XYZ", value=False, key="relax_savexyz")
    stride = st.number_input("Save every Nth step", value=default_stride, min_value=1, max_value=1000,
                             step=1, key="relax_stride")

    if pmg_obj is None:
        st.info("Upload a structure to optimize.")
        return
    if isinstance(pmg_obj, Structure):
        st.caption("Initial lattice: " + _lattice_summary(pmg_obj))

    if not st.button("Run optimization", type="primary", key="relax_run"):
        return

    try:
        st.session_state.stop_requested = False
        atoms = atoms_from(pmg_obj)
        atoms.calc = get_calculator(model_family, variant)

        target = atoms
        if mode == "Variable cell (shape + volume)":
            target = UnitCellFilter(atoms)
        elif mode == "Cell shape only (approx. fixed volume)":
            target = StrainFilter(atoms, mask=[0,0,0,1,1,1])

        progress = st.progress(0, text="Starting optimization…")
        pct_label = st.empty()
        energies: list[float] = []
        buf_xyz = io.StringIO() if save_xyz else None
        step_counter = {"i": 0}

        def on_step():
            if st.session_state.stop_requested:
                raise RuntimeError("Stop requested by user.")
            step_counter["i"] += 1
            i = step_counter["i"]
            e = atoms.get_potential_energy(); energies.append(float(e))
            pct = min(int(100 * i / max_steps), 99)
            progress.progress(pct, text=f"Optimizing… {pct}%"); pct_label.write(f"**Progress:** {pct}%")
            if buf_xyz is not None and (i % int(st.session_state.get("relax_stride", default_stride)) == 0):
                ase_write(buf_xyz, atoms.copy(), format="xyz")

        dyn = _make_optimizer(optimizer, target)
        dyn.attach(on_step, interval=1)
        dyn.run(fmax=float(fmax), steps=int(max_steps))

        final_e = atoms.get_potential_energy()
        fmax_val = float(np.abs(atoms.get_forces()).max())
        progress.progress(100, text="Done"); pct_label.write("**Progress:** 100%")

        final_struct = AseAtomsAdaptor.get_structure(atoms)
        if isinstance(pmg_obj, Structure):
            st.write("**Final lattice:** " + _lattice_summary(final_struct))
        st.success(f"Relaxed: E = {final_e:.6f} eV | max|F| = {fmax_val:.6f} eV/Å")

        # persist artifacts to session state (so they survive reruns/downloads)
        st.session_state[_PLOT_KEY] = _plot_energy(energies)
        if st.session_state[_PLOT_KEY]:
            st.image(st.session_state[_PLOT_KEY], caption="Energy vs optimization step", use_container_width=False)

        st.session_state[_CIF_KEY] = final_struct.to(fmt="cif").encode()
        st.download_button("⬇️ Download relaxed CIF", st.session_state[_CIF_KEY],
                           file_name="relaxed_structure.cif", key="relax_dl_cif2")

        if buf_xyz is not None:
            st.session_state[_XYZ_KEY] = buf_xyz.getvalue().encode()
            st.download_button("⬇️ Download trajectory (XYZ)", st.session_state[_XYZ_KEY],
                               file_name="relax_trajectory.xyz", key="relax_dl_xyz2")

    except Exception as exc:
        st.error(f"Optimization failed: {exc}")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
