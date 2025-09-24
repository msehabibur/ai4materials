from __future__ import annotations
import io, traceback
import numpy as np
import streamlit as st
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.io import write as ase_write
from pymatgen.io.ase import AseAtomsAdaptor
from core.struct import atoms_from
from core.utils import msd_and_plot
from matgl.ext.ase import M3GNetCalculator

def md_tab(pmg_obj, potential):
    st.subheader("Molecular Dynamics (NVT via Langevin)")

    col1, col2, col3 = st.columns(3)
    with col1:
        T = st.number_input("Temperature (K)", value=300, min_value=1, max_value=3000, step=10)
    with col2:
        dt_fs = st.number_input("Timestep (fs)", value=1.0, min_value=0.1, max_value=10.0, step=0.1, format="%.1f")
    with col3:
        steps = st.number_input("MD steps", value=2000, min_value=10, max_value=200000, step=100)

    friction = st.number_input("Friction γ (1/ps)", value=1.0, min_value=0.01, max_value=100.0, step=0.1, format="%.2f")
    save_xyz = st.checkbox("Save trajectory (XYZ)", value=True)

    run = st.button("Run MD", type="primary", disabled=(pmg_obj is None or potential is None))
    if not run:
        return

    try:
        atoms = atoms_from(pmg_obj)
        atoms.calc = M3GNetCalculator(potential=potential)

        MaxwellBoltzmannDistribution(atoms, temperature_K=float(T))

        gamma_fs = float(friction) / 1000.0  # 1/ps -> 1/fs
        dyn = Langevin(atoms, timestep=float(dt_fs) * units.fs, temperature_K=float(T), friction=gamma_fs)

        positions = []
        def _snap():
            positions.append(atoms.get_positions().copy())
        dyn.attach(_snap, interval=1)
        dyn.run(int(steps))

        positions = np.array(positions)
        png = msd_and_plot(positions, float(dt_fs))
        st.image(png, caption="MSD vs Time", use_container_width=True)

        # downloads
        if save_xyz:
            buf = io.StringIO()
            for pos in positions:
                tmp = atoms.copy()
                tmp.set_positions(pos)
                ase_write(buf, tmp, format="xyz")
            st.download_button("⬇️ Download MD trajectory (XYZ)", buf.getvalue().encode(), file_name="md_trajectory.xyz")

        atoms.set_positions(positions[-1])
        final_struct = AseAtomsAdaptor.get_structure(atoms)
        st.download_button("⬇️ Download last frame (CIF)", final_struct.to(fmt="cif").encode(), file_name="md_last_frame.cif")

    except Exception as exc:
        st.error("MD failed.")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
