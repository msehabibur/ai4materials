from __future__ import annotations
import os, io, tempfile, traceback
import numpy as np
import streamlit as st
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.io import write as ase_write
from pymatgen.io.ase import AseAtomsAdaptor
from core.struct import atoms_from
from matgl.ext.ase import M3GNetCalculator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _online_msd_update(state, r):
    """
    Online MSD relative to first frame r0; no need to store all frames.
    state: dict with keys r0, t, msd_sum
    r   : (n_atoms,3) current positions
    """
    if state["r0"] is None:
        state["r0"] = r.copy()
        return
    dr = r - state["r0"]
    msd_frame = float((dr * dr).sum(axis=1).mean())  # average over atoms
    state["t"].append(state["t"][-1] + 1 if state["t"] else 0)
    state["msd_sum"].append(msd_frame)

def _plot_msd(msd_vals, dt_fs):
    if not msd_vals:
        return None
    t_ps = np.arange(len(msd_vals)) * float(dt_fs) / 1000.0
    fig = plt.figure()
    plt.plot(t_ps, msd_vals)
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD (Å$^2$)")
    plt.title("Mean Squared Displacement")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def md_tab(pmg_obj, potential):
    st.subheader("Molecular Dynamics (NVT via Langevin) — memory-light")

    col1, col2, col3 = st.columns(3)
    with col1:
        T = st.number_input("Temperature (K)", value=300, min_value=1, max_value=3000, step=10)
    with col2:
        dt_fs = st.number_input("Timestep (fs)", value=1.0, min_value=0.1, max_value=10.0, step=0.1, format="%.1f")
    with col3:
        steps = st.number_input("MD steps", value=2000, min_value=10, max_value=200000, step=100)

    col4, col5, col6 = st.columns(3)
    with col4:
        friction = st.number_input("Friction γ (1/ps)", value=1.0, min_value=0.01, max_value=100.0, step=0.1, format="%.2f")
    with col5:
        save_xyz = st.checkbox("Save trajectory (XYZ to temp file)", value=False)
    with col6:
        stride = st.number_input("Frame stride (save every Nth)", value=10, min_value=1, max_value=1000, step=1)

    run = st.button("Run MD", type="primary", disabled=(pmg_obj is None or potential is None))
    if not run:
        return

    try:
        atoms = atoms_from(pmg_obj)
        atoms.calc = M3GNetCalculator(potential=potential)

        MaxwellBoltzmannDistribution(atoms, temperature_K=float(T))

        gamma_fs = float(friction) / 1000.0  # 1/ps -> 1/fs
        dyn = Langevin(atoms, timestep=float(dt_fs) * units.fs, temperature_K=float(T), friction=gamma_fs)

        # Online MSD state (no big arrays)
        state = {"r0": None, "t": [], "msd_sum": []}

        # Optional streaming trajectory file
        tmp_path = None
        xyz_file = None
        if save_xyz:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
            tmp_path = tmp.name
            tmp.close()  # reopen in text mode
            xyz_file = open(tmp_path, "w")

        def _on_step(i=[0]):
            i[0] += 1
            pos = atoms.get_positions()
            _online_msd_update(state, pos)

            if save_xyz and (i[0] % int(stride) == 0):
                # Write a single frame incrementally
                buf = io.StringIO()
                tmp_atoms = atoms.copy()
                tmp_atoms.set_positions(pos)
                ase_write(buf, tmp_atoms, format="xyz")
                xyz_file.write(buf.getvalue())

        dyn.attach(_on_step, interval=1)
        dyn.run(int(steps))

        if xyz_file is not None:
            xyz_file.flush()
            xyz_file.close()

        # Plot MSD
        png = _plot_msd(state["msd_sum"], float(dt_fs))
        if png is not None:
            st.image(png, caption="MSD vs Time", use_container_width=True)

        # Last frame → CIF
        final_struct = AseAtomsAdaptor.get_structure(atoms)
        st.download_button("⬇️ Download last frame (CIF)", final_struct.to(fmt="cif").encode(), file_name="md_last_frame.cif")

        if tmp_path:
            with open(tmp_path, "rb") as fh:
                st.download_button("⬇️ Download streamed trajectory (XYZ)", fh.read(), file_name="md_trajectory.xyz")
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except Exception as exc:
        st.error("MD failed.")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
