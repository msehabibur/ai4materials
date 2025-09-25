from __future__ import annotations
import os, io, tempfile, traceback, numpy as np
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _online_msd_state(): return {"r0": None, "msd": []}
def _online_msd_update(state, r):
    if state["r0"] is None:
        state["r0"] = r.copy(); state["msd"].append(0.0); return
    dr = r - state["r0"]; state["msd"].append(float((dr*dr).sum(axis=1).mean()))

def _plot_msd(msd_vals, dt_fs):
    if not msd_vals: return None
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(6.0, 4.5))
    t_ps = np.arange(len(msd_vals)) * float(dt_fs) / 1000.0
    plt.plot(t_ps, msd_vals, linewidth=2)
    plt.xlabel("Time (ps)"); plt.ylabel("MSD (Å$^2$)"); plt.title("Mean Squared Displacement")
    buf = io.BytesIO(); fig.savefig(buf, dpi=180, bbox_inches="tight"); plt.close(fig)
    buf.seek(0); return buf.read()

def md_tab(pmg_obj, chgnet_variant: str):
    st.subheader("Molecular Dynamics — CHGNet")

    c0, c00 = st.columns([1,1])
    with c0:
        ensemble = st.selectbox("Ensemble", ["NVE", "NVT (Langevin)", "NPT (Berendsen)"], key="md_ensemble")
    with c00:
        if ensemble.startswith("NPT") and NPTBerendsen is None:
            st.warning("NPT Berendsen integrator unavailable here; pick NVE or NVT.", icon="⚠️")

    c1, c2, c3 = st.columns(3)
    with c1:
        T = st.number_input("Temperature (K)", value=300, min_value=1, max_value=3000, step=10, key="md_T")
    with c2:
        dt_fs = st.number_input("Timestep (fs)", value=1.0, min_value=0.1, max_value=10.0, step=0.1,
                                format="%.1f", key="md_dt")
    with c3:
        steps = st.number_input("MD steps", value=2000, min_value=10, max_value=200000, step=100, key="md_steps")

    c4, c5, c6 = st.columns(3)
    with c4:
        friction = st.number_input("Friction γ (1/ps) [NVT]", value=1.0, min_value=0.01, max_value=100.0,
                                   step=0.1, format="%.2f", key="md_gamma")
    with c5:
        pressure_bar = st.number_input("Pressure (bar) [NPT]", value=1.0, min_value=0.0, max_value=100.0,
                                       step=0.5, key="md_pbar")
    with c6:
        save_xyz = st.checkbox("Stream trajectory to XYZ", value=False, key="md_savexyz")
    stride = st.number_input("Save every Nth step", value=25, min_value=1, max_value=1000, step=1, key="md_stride")

    if pmg_obj is None:
        st.info("Upload a structure to run MD.")
        return
    if not st.button("Run MD", type="primary", key="md_run"):
        return

    try:
        st.session_state.stop_requested = False
        atoms = atoms_from(pmg_obj)
        atoms.calc = get_chgnet_calculator(chgnet_variant)
        MaxwellBoltzmannDistribution(atoms, temperature_K=float(T))

        dt = float(dt_fs) * units.fs
        if ensemble.startswith("NVE"):
            dyn = VelocityVerlet(atoms, dt)
        elif ensemble.startswith("NVT"):
            gamma_fs = float(friction) / 1000.0
            dyn = Langevin(atoms, timestep=dt, temperature_K=float(T), friction=gamma_fs)
        else:
            if NPTBerendsen is None:
                st.error("NPT not available. Choose NVE or NVT.")
                return
            p_target = float(pressure_bar) * 1e5
            dyn = NPTBerendsen(atoms, timestep=dt, temperature_K=float(T),
                               taut=100.0*units.fs, pressure=p_target, compressibility=4.5e-10)

        progress = st.progress(0, text="Starting MD…")
        msd_state = _online_msd_state()

        tmp_path, xyz_file = (None, None)
        if save_xyz:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
            tmp_path = tmp.name; tmp.close()
            st.session_state.tmp_paths.append(tmp_path)
            xyz_file = open(tmp_path, "w")

        step_idx = {"i": 0}
        def on_step():
            if st.session_state.stop_requested:
                raise RuntimeError("Stop requested by user.")
            step_idx["i"] += 1; i = step_idx["i"]
            pos = atoms.get_positions(); _online_msd_update(msd_state, pos)
            if save_xyz and (i % int(st.session_state.get("md_stride", 25)) == 0):
                s = io.StringIO(); frame = atoms.copy(); frame.set_positions(pos)
                ase_write(s, frame, format="xyz"); xyz_file.write(s.getvalue())
            pct = min(int(100 * i / steps), 99); progress.progress(pct, text=f"MD running… {pct}%")

        dyn.attach(on_step, interval=1)
        dyn.run(int(steps))

        if xyz_file is not None:
            xyz_file.flush(); xyz_file.close()

        progress.progress(100, text="Done")

        png = _plot_msd(msd_state["msd"], float(dt_fs))
        if png: st.image(png, caption="MSD vs time", use_container_width=False)

        final_struct = AseAtomsAdaptor.get_structure(atoms)
        st.download_button("⬇️ Download last frame (CIF)", final_struct.to(fmt="cif").encode(),
                           file_name="md_last_frame.cif", key="md_dl_cif")
        if tmp_path:
            with open(tmp_path, "rb") as fh:
                st.download_button("⬇️ Download trajectory (XYZ, streamed)", fh.read(),
                                   file_name="md_trajectory.xyz", key="md_dl_xyz")

    except Exception as exc:
        st.error(f"MD failed: {exc}")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
