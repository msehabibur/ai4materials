from __future__ import annotations
import io, traceback
import numpy as np
import streamlit as st
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import BFGS
from ase.io import write as ase_write
from core.struct import atoms_from

# M3GNet ASE calculator is constructed in app via potential
from matgl.ext.ase import M3GNetCalculator  # import here to fail-fast nicely

def relax_tab(pmg_obj, potential):
    st.subheader("Geometry Optimization (BFGS on M3GNet PES)")

    col1, col2 = st.columns(2)
    with col1:
        fmax = st.number_input("Force convergence (eV/Å)", value=0.05, min_value=0.001, max_value=1.0, step=0.01, format="%.3f")
    with col2:
        max_steps = st.number_input("Max steps", value=200, min_value=10, max_value=5000, step=10)
    record_traj = st.checkbox("Record trajectory (XYZ)", value=True)

    run = st.button("Run Relaxation", type="primary", disabled=(pmg_obj is None or potential is None))
    if not run:
        return

    try:
        atoms = atoms_from(pmg_obj)
        atoms.calc = M3GNetCalculator(potential=potential)

        frames = []
        def _rec():
            frames.append(atoms.get_positions().copy())

        dyn = BFGS(atoms, logfile=None, maxstep=0.2)
        dyn.attach(_rec, interval=1)
        dyn.run(fmax=float(fmax), steps=int(max_steps))

        e = atoms.get_potential_energy()
        fmax_val = float(np.abs(atoms.get_forces()).max())
        st.success(f"Done. E = {e:.6f} eV | max|F| = {fmax_val:.4f} eV/Å")

        # downloads
        final_struct = AseAtomsAdaptor.get_structure(atoms)
        st.download_button("⬇️ Download relaxed CIF", final_struct.to(fmt="cif").encode(), file_name="relaxed_structure.cif")

        if record_traj and frames:
            from ase.io import write as ase_write
            import io
            buf = io.StringIO()
            for pos in frames:
                tmp = atoms.copy()
                tmp.set_positions(pos)
                ase_write(buf, tmp, format="xyz")
            st.download_button("⬇️ Download relaxation trajectory (XYZ)", buf.getvalue().encode(), file_name="relaxation_trajectory.xyz")

    except Exception as exc:
        st.error("Relaxation failed.")
        with st.expander("Traceback"):
            import traceback
            st.code("".join(traceback.format_exception(exc)))
