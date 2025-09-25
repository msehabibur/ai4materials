from __future__ import annotations
import os, io, tempfile, traceback
import numpy as np
import streamlit as st
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
try:
    from ase.md.nptberendsen import NPTBerendsen
except Exception:
    NPTBerendsen = None
from ase.io import write as ase_write
from pymatgen.io.ase import AseAtomsAdaptor
from core.struct import atoms_from
from core.model import get_chgnet_calculator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _online_msd_state():
    return {"r0": None, "msd": []}

def _online_msd_update(state, r):
    if state["r0"] is None:
        state["r0"] = r.copy()
        state["msd"].append(0.0)
        return
    dr = r - state["r0"]
    state["msd"].append(float((dr * dr).sum(axis=1).mean()))

def _plot_msd(msd_vals, dt_fs):
    if not msd_vals:
        return None
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(6.0, 4.5))
    t_ps = np.arange(len(msd_vals)) * float(dt_fs) / 1000.0
    plt.plot(t_ps, msd_vals, linewidth=2)
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD (Å$^2$)")
    plt.title("Mean Squared Displacement")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def md_tab(pmg_obj):
    st.subheader("Molecular Dynamics — CHGNet")

    col0, col00 = st.columns([1,1])
    with col0:
        ensemble = st.selectbox("Ensemble", ["NVE", "NVT (Langevin)", "NPT (Berendsen)"])
    with col00:
        if ensemble.startswith("NPT") and NPTBerendsen is None:
            st.warning("NPT Berendsen integrator unavailable in this environment; choose NVE or NVT.")

    col1, col2, col3 = st.columns(3)
    with col1:
        T = st.number_input("Temperature (K)", value=300, min_value=1, max_value=3000, step=10)
    with col2:
        dt_fs = st.number_input("Timestep (fs)", value=1.0, min_value=0.1, max_value=10.0, step=0.1, format="%.1f")
    with col3:
        steps = st.number_input("MD steps", value=2000, min_value=10, max_value=200000, step=100)

    col4, col5, col6 = st.columns(3)
    with col4:
        friction = st.number_input("Friction γ (1/ps) [NVT only]", value=1.0, min_value=0.01, max_value=100.0, step=0.1, format="%.2f")
    with col5:
        pressure_bar = st.number_input("Pressure (bar) [NPT only]", value=1.0, min_value=0.0, max_value=100.0, step=0.5)
    with col6:
        save_xyz = st.checkbox("Stream trajectory to XYZ", value=False)
    stride = st.number_input("Save every Nth step", value=25, min_value=1, max_value=1000, step=1)

    if pmg_obj is None:
        st.info("Upload a structure to run MD.")
        return

    run = st.button("Run MD", type="primary")
    if not run:
        return

    try:
        st.session_state.stop_requested = False
        atoms = atoms_from(pmg_obj)
        atoms.calc = get_chgnet_calculator()
        MaxwellBoltzmannDistribution(atoms, temperature_K=float(T))

        dt = float(dt_fs) * units.fs
        if ensemble.startswith("NVE"):
            dyn = VelocityVerlet(atoms, dt)
        elif ensemble.startswith("NVT"):
            gamma_fs = float(friction) / 1000.0
            dyn = Langevin(atoms, timestep=dt, temperature_K=float(T), friction=gamma_fs)
        else:
            if NPTBerendsen is None:
                st.error("NPT Berendsen not available. Choose NVE or NVT.")
                return
            # pressure in bar -> Pa
            p_target = float(pressure_bar) * 1e5
            dyn = NPTBerendsen(
                atoms, timestep=dt, temperature_K=float(T),
                taut=100.0*units.fs, pressure=p_target, compressibility=4.5e-10
            )

        progress = st.progress(0, text="Starting MD…")
        msd_state = _online_msd_state()

        tmp_path, xyz_file = (None, None)
        if save_xyz:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
            tmp_path = tmp.name
            tmp.close()
            st.session_state.tmp_paths.append(tmp_path)
            xyz_file = open(tmp_path, "w")

        step_idx = {"i": 0}
        def on_step():
            if st.session_state.stop_requested:
                raise RuntimeError("Stop requested by user.")
            step_idx["i"] += 1
            i = step_idx["i"]
            pos = atoms.get_positions()
            _online_msd_update(msd_state, pos)
            if save_xyz and (i % int(stride) == 0):
                s = io.StringIO()
                frame = atoms.copy()
                frame.set_positions(pos)
                ase_write(s, frame, format="xyz")
                xyz_file.write(s.getvalue())
            pct = min(int(100 * i / steps), 99)
            progress.progress(pct, text=f"MD running… {pct}%")

        dyn.attach(on_step, interval=1)
        dyn.run(int(steps))

        if xyz_file is not None:
            xyz_file.flush(); xyz_file.close()

        progress.progress(100, text="Done")

        png = _plot_msd(msd_state["msd"], float(dt_fs))
        if png is not None:
            st.image(png, caption="MSD vs time", use_container_width=False)

        final_struct = AseAtomsAdaptor.get_structure(atoms)
        st.download_button("⬇️ Download last frame (CIF)", final_struct.to(fmt="cif").encode(), file_name="md_last_frame.cif")
        if tmp_path:
            with open(tmp_path, "rb") as fh:
                st.download_button("⬇️ Download trajectory (XYZ, streamed)", fh.read(), file_name="md_trajectory.xyz")

    except Exception as exc:
        st.error(f"MD failed: {exc}")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
