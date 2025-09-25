# tabs/md_tab.py
from __future__ import annotations
import io, os, tempfile, traceback
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

from ase.io import write as ase_write, read as ase_read
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor

from core.struct import atoms_from
from core.model import get_calculator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

_MSD_PLOT_KEY = "md_msd_png"
_MD_LAST_CIF  = "md_last_cif"
_MD_TRAJ      = "md_traj_ase"
_MD_XYZ       = "md_traj_xyz"

def _plot_msd(msd_vals, dt_fs: float) -> bytes | None:
    if not msd_vals: return None
    plt.rcParams.update({"font.size": 16})
    fig = plt.figure(figsize=(5.0, 3.6))
    ax = plt.gca()
    sf = ScalarFormatter(useMathText=False); sf.set_scientific(False); sf.set_useOffset(False)
    ax.yaxis.set_major_formatter(sf)
    t_ps = np.arange(len(msd_vals)) * float(dt_fs) / 1000.0
    ax.plot(t_ps, msd_vals, linewidth=2)
    ax.set_xlabel("Time (ps)"); ax.set_ylabel("MSD (Å$^2$)")
    ax.set_title("Mean Squared Displacement")
    buf = io.BytesIO(); fig.savefig(buf, dpi=160, bbox_inches="tight")
    plt.close(fig); buf.seek(0)
    return buf.read()

def md_tab(pmg_obj, _family="MACE", _variant="default"):
    st.subheader("Molecular Dynamics — MACE")
    if st.session_state.get(_MSD_PLOT_KEY):
        st.image(st.session_state[_MSD_PLOT_KEY], caption="MSD vs time")
    if st.session_state.get(_MD_LAST_CIF):
        st.download_button("⬇️ Last frame (CIF)", st.session_state[_MD_LAST_CIF], "md_last_frame.cif", key="md_dl_cif_persist")
    if st.session_state.get(_MD_TRAJ):
        st.download_button("⬇️ ASE trajectory (.traj)", st.session_state[_MD_TRAJ], "md_trajectory.traj", key="md_dl_traj_persist")
    if st.session_state.get(_MD_XYZ):
        st.download_button("⬇️ Trajectory (XYZ)", st.session_state[_MD_XYZ], "md_trajectory.xyz", key="md_dl_xyz_persist")

    if pmg_obj is None:
        st.info("Upload/select a structure in the viewer to run MD.")
        return

    c0, c00 = st.columns(2)
    with c0:
        ensemble = st.selectbox("Ensemble", ["NVE", "NVT (Langevin)", "NPT (Berendsen)"], key="md_ensemble")
    with c00:
        if ensemble.startswith("NPT") and NPTBerendsen is None:
            st.warning("NPT Berendsen unavailable in this environment.", icon="⚠️")

    cT1, cT2, cT3 = st.columns(3)
    with cT1:
        T_init = st.number_input("Initial T (K)", value=300, min_value=1, max_value=5000, step=10, key="md_T_init")
    with cT2:
        T_target = st.number_input("Target T (K)", value=300, min_value=1, max_value=5000, step=10, key="md_T_target")
    with cT3:
        dt_fs = st.number_input("Timestep (fs)", value=1.0, min_value=0.1, max_value=10.0, step=0.1, format="%.1f", key="md_dt")

    c1, c2, c3 = st.columns(3)
    with c1:
        steps = st.number_input("MD steps", value=2000, min_value=10, max_value=200000, step=100, key="md_steps")
    with c2:
        friction = st.number_input("Friction γ (1/ps) [NVT]", value=1.0, min_value=0.01, max_value=100.0, step=0.1, format="%.2f", key="md_gamma")
    with c3:
        pressure_bar = st.number_input("Pressure (bar) [NPT]", value=1.0, min_value=0.0, max_value=100.0, step=0.5, key="md_pbar")

    c4, c5 = st.columns(2)
    with c4:
        stride = st.number_input("Save every Nth step", value=25, min_value=1, max_value=1000, step=1, key="md_stride")
    with c5:
        save_xyz = st.checkbox("Also write EXTXYZ (downloadable)", value=True, key="md_savexyz")

    if not st.button("Run MD", type="primary", key="md_run"):
        return

    try:
        atoms = atoms_from(pmg_obj)
        atoms.calc = get_calculator(_family, _variant)
        from ase.md import MDLogger  # optional textual log if needed

        MaxwellBoltzmannDistribution(atoms, temperature_K=float(T_init))
        dt = float(dt_fs) * units.fs

        if ensemble.startswith("NVE"):
            dyn = VelocityVerlet(atoms, dt)
        elif ensemble.startswith("NVT"):
            gamma_fs = float(friction) / 1000.0
            dyn = Langevin(atoms, timestep=dt, temperature_K=float(T_target), friction=gamma_fs)
        else:
            if NPTBerendsen is None:
                st.error("NPT not available. Choose NVE or NVT.")
                return
            p_target = float(pressure_bar) * 1e5
            dyn = NPTBerendsen(atoms, timestep=dt, temperature_K=float(T_target),
                               taut=100.0 * units.fs, pressure=p_target, compressibility=4.5e-10)

        progress = st.progress(0, text="Starting MD…")
        pct_label = st.empty()
        msd_state = {"r0": None, "msd": []}

        traj_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".traj"); traj_path = traj_tmp.name; traj_tmp.close()
        traj_writer = Trajectory(traj_path, "w", atoms); traj_writer.write(atoms)

        xyz_path = None
        if save_xyz:
            xyz_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz"); xyz_path = xyz_tmp.name; xyz_tmp.close()
            ase_write(xyz_path, atoms, format="extxyz", append=False)

        idx = {"i": 0}
        def on_step():
            idx["i"] += 1; i = idx["i"]
            R = atoms.get_positions()
            if msd_state["r0"] is None:
                msd_state["r0"] = R.copy(); msd_state["msd"].append(0.0)
            else:
                d = R - msd_state["r0"]
                msd_state["msd"].append(float((d*d).sum(axis=1).mean()))
            if i % int(stride) == 0:
                traj_writer.write(atoms)
                if xyz_path: ase_write(xyz_path, atoms, format="extxyz", append=True)
            pct = min(int(100 * i / int(steps)), 99)
            progress.progress(pct, text=f"MD running… {pct}%"); pct_label.write(f"**Progress:** {pct}%")

        dyn.attach(on_step, interval=1)
        dyn.run(int(steps))
        try: traj_writer.close()
        except Exception: pass

        progress.progress(100, text="Done"); pct_label.write("**Progress:** 100%")

        # Persist artifacts
        png = _plot_msd(msd_vals=msd_state["msd"], dt_fs=float(dt_fs))
        if png: st.session_state[_MSD_PLOT_KEY] = png; st.image(png, caption="MSD vs time")
        final_struct = AseAtomsAdaptor.get_structure(atoms)
        st.session_state[_MD_LAST_CIF] = final_struct.to(fmt="cif").encode()
        st.download_button("⬇️ Last frame (CIF)", st.session_state[_MD_LAST_CIF], "md_last_frame.cif", key="md_dl_cif")

        with open(traj_path, "rb") as fh:
            st.session_state[_MD_TRAJ] = fh.read()
        st.download_button("⬇️ ASE trajectory (.traj)", st.session_state[_MD_TRAJ], "md_trajectory.traj", key="md_dl_traj")

        if xyz_path and os.path.exists(xyz_path):
            with open(xyz_path, "rb") as fh:
                st.session_state[_MD_XYZ] = fh.read()
            st.download_button("⬇️ Trajectory (XYZ)", st.session_state[_MD_XYZ], "md_trajectory.xyz", key="md_dl_xyz")

    except Exception as exc:
        st.error(f"MD failed: {exc}")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
