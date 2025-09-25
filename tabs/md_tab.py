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
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import ScalarFormatter

# ---- Session keys for persisted artifacts (so they don't vanish after downloads) ----
_MSD_PLOT_KEY = "md_msd_png"
_MD_LAST_CIF  = "md_last_cif"
_MD_XYZ       = "md_traj_xyz"
_MD_GIF       = "md_traj_gif"
_MD_TRAJ      = "md_traj_ase"  # .traj bytes for download

def _ensure_tmp_list():
    if "tmp_paths" not in st.session_state:
        st.session_state.tmp_paths = []

# ---- Online MSD helpers ----
def _online_msd_state(): return {"r0": None, "msd": []}

def _online_msd_update(state, r: np.ndarray):
    if state["r0"] is None:
        state["r0"] = r.copy()
        state["msd"].append(0.0)
        return
    dr = r - state["r0"]
    state["msd"].append(float((dr * dr).sum(axis=1).mean()))

def _plot_msd(msd_vals, dt_fs: float) -> bytes | None:
    if not msd_vals:
        return None
    plt.rcParams.update({"font.size": 18})
    fig = plt.figure(figsize=(5.0, 3.6))
    ax = plt.gca()
    sf = ScalarFormatter(useMathText=False)
    sf.set_scientific(False)
    sf.set_useOffset(False)
    ax.yaxis.set_major_formatter(sf)
    t_ps = np.arange(len(msd_vals)) * float(dt_fs) / 1000.0
    ax.plot(t_ps, msd_vals, linewidth=2)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("MSD (Å$^2$)")
    ax.set_title("Mean Squared Displacement")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def _make_traj_gif_from_frames(frames, forced_symbols: str | None = None) -> tuple[bytes, int]:
    """Build an animated GIF from a list of ASE Atoms frames. Returns (gif_bytes, n_frames)."""
    try:
        if forced_symbols:
            for a in frames:
                a.set_chemical_symbols(forced_symbols)
        n = len(frames)
        if n < 2:
            return b"", n

        fig, ax = plt.subplots(figsize=(4, 4))
        from ase.visualize.plot import plot_atoms

        def update(i):
            ax.clear()
            plot_atoms(frames[i], ax, radii=0.8, rotation=("45x,45y,0z"))
            ax.set_title(f"Frame {i + 1}")
            ax.axis("off")

        ani = FuncAnimation(fig, update, frames=n, interval=200)
        buf = io.BytesIO()
        ani.save(buf, writer="pillow", dpi=200, fps=60)
        plt.close(fig)
        buf.seek(0)
        return buf.read(), n
    except Exception:
        return b"", 0

def md_tab(pmg_obj, model_family: str, variant: str):
    st.subheader("Molecular Dynamics")
    _ensure_tmp_list()

    # ---- Persisted artifacts (remain visible across reruns/downloads) ----
    if st.session_state.get(_MSD_PLOT_KEY):
        st.image(st.session_state[_MSD_PLOT_KEY], caption="MSD vs time", use_container_width=False)
    if st.session_state.get(_MD_GIF):
        st.image(st.session_state[_MD_GIF], caption="Trajectory animation (GIF)", use_container_width=False)
    if st.session_state.get(_MD_LAST_CIF):
        st.download_button("⬇️ Download last frame (CIF)", st.session_state[_MD_LAST_CIF],
                           "md_last_frame.cif", key="md_dl_cif_persist")
    if st.session_state.get(_MD_TRAJ):
        st.download_button("⬇️ Download ASE trajectory (.traj)", st.session_state[_MD_TRAJ],
                           "md_trajectory.traj", key="md_dl_traj_persist")
    if st.session_state.get(_MD_XYZ):
        st.download_button("⬇️ Download trajectory (XYZ)", st.session_state[_MD_XYZ],
                           "md_trajectory.xyz", key="md_dl_xyz_persist")

    # ---- Controls ----
    c0, c00 = st.columns([1, 1])
    with c0:
        ensemble = st.selectbox("Ensemble", ["NVE", "NVT (Langevin)", "NPT (Berendsen)"], key="md_ensemble")
    with c00:
        if ensemble.startswith("NPT") and NPTBerendsen is None:
            st.warning("NPT Berendsen unavailable; choose NVE or NVT.", icon="⚠️")

    cT1, cT2, cT3 = st.columns(3)
    with cT1:
        T_init = st.number_input("Initial temperature (K)", value=300, min_value=1, max_value=5000, step=10, key="md_T_init")
    with cT2:
        T_target = st.number_input("Target temperature (K)", value=300, min_value=1, max_value=5000, step=10, key="md_T_target")
    with cT3:
        dt_fs = st.number_input("Timestep (fs)", value=1.0, min_value=0.1, max_value=10.0, step=0.1, format="%.1f", key="md_dt")

    c1, c2, c3 = st.columns(3)
    with c1:
        steps = st.number_input("MD steps", value=2000, min_value=10, max_value=200000, step=100, key="md_steps")
    with c2:
        friction = st.number_input("Friction γ (1/ps) [NVT]", value=1.0, min_value=0.01, max_value=100.0, step=0.1, format="%.2f", key="md_gamma")
    with c3:
        pressure_bar = st.number_input("Pressure (bar) [NPT]", value=1.0, min_value=0.0, max_value=100.0, step=0.5, key="md_pbar")

    c4, c5, c6 = st.columns(3)
    with c4:
        save_xyz = st.checkbox("Stream trajectory to XYZ (downloadable)", value=False, key="md_savexyz")
    with c5:
        stride = st.number_input("Save every Nth step", value=25, min_value=1, max_value=1000, step=1, key="md_stride")
    with c6:
        make_gif = st.checkbox("Render GIF animation", value=False, key="md_makegif")
        st.caption("GIF is built from a robust internal .traj stream (independent of XYZ).")

    if pmg_obj is None:
        st.info("Upload a structure to run MD.")
        return

    if not st.button("Run MD", type="primary", key="md_run"):
        return

    try:
        st.session_state.stop_requested = False

        # ---- Build atoms + calculator ----
        atoms = atoms_from(pmg_obj)
        atoms.calc = get_calculator(model_family, variant)

        # ---- Initialize velocities ----
        MaxwellBoltzmannDistribution(atoms, temperature_K=float(T_init))

        # ---- Choose integrator ----
        dt = float(dt_fs) * units.fs
        if ensemble.startswith("NVE"):
            dyn = VelocityVerlet(atoms, dt)
        elif ensemble.startswith("NVT"):
            gamma_fs = float(friction) / 1000.0  # 1/ps -> 1/fs
            dyn = Langevin(atoms, timestep=dt, temperature_K=float(T_target), friction=gamma_fs)
        else:
            if NPTBerendsen is None:
                st.error("NPT not available. Choose NVE or NVT.")
                return
            p_target = float(pressure_bar) * 1e5  # bar -> Pa
            dyn = NPTBerendsen(
                atoms, timestep=dt, temperature_K=float(T_target),
                taut=100.0 * units.fs, pressure=p_target, compressibility=4.5e-10
            )

        # ---- Progress + MSD state ----
        progress = st.progress(0, text="Starting MD…")
        pct_label = st.empty()
        msd_state = _online_msd_state()

        # ---- Trajectory writers ----
        # Robust internal writer: ASE .traj (always used if we need GIF or XYZ)
        need_frames = make_gif or save_xyz
        traj_path = None
        traj_writer = None
        if need_frames:
            traj_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".traj")
            traj_path = traj_tmp.name
            traj_tmp.close()
            st.session_state.tmp_paths.append(traj_path)
            traj_writer = Trajectory(traj_path, "w", atoms)
            traj_writer.write(atoms)  # initial frame

        # Optional external XYZ stream for user download
        xyz_path = None
        if save_xyz:
            xyz_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
            xyz_path = xyz_tmp.name
            xyz_tmp.close()
            st.session_state.tmp_paths.append(xyz_path)
            # initial frame in EXTXYZ (more robust than plain xyz)
            ase_write(xyz_path, atoms, format="extxyz", append=False)

        # ---- Callback each MD step ----
        step_idx = {"i": 0}
        def on_step():
            if st.session_state.stop_requested:
                raise RuntimeError("Stop requested by user.")
            step_idx["i"] += 1
            i = step_idx["i"]

            pos = atoms.get_positions()
            _online_msd_update(msd_state, pos)

            if need_frames and (i % int(st.session_state.get("md_stride", 25)) == 0):
                traj_writer.write(atoms)
                if save_xyz:
                    ase_write(xyz_path, atoms, format="extxyz", append=True)

            pct = min(int(100 * i / steps), 99)
            progress.progress(pct, text=f"MD running… {pct}%")
            pct_label.write(f"**Progress:** {pct}%")

        dyn.attach(on_step, interval=1)

        # ---- Run MD ----
        dyn.run(int(steps))

        # ---- Close trajectory writer ----
        if traj_writer is not None:
            try: traj_writer.close()
            except Exception: pass

        progress.progress(100, text="Done")
        pct_label.write("**Progress:** 100%")

        # ---- Persist MSD plot ----
        st.session_state[_MSD_PLOT_KEY] = _plot_msd(msd_state["msd"], float(dt_fs))
        if st.session_state[_MSD_PLOT_KEY]:
            st.image(st.session_state[_MSD_PLOT_KEY], caption="MSD vs time", use_container_width=False)

        # ---- Persist last frame (CIF) ----
        final_struct = AseAtomsAdaptor.get_structure(atoms)
        st.session_state[_MD_LAST_CIF] = final_struct.to(fmt="cif").encode()
        st.download_button("⬇️ Download last frame (CIF)", st.session_state[_MD_LAST_CIF],
                           file_name="md_last_frame.cif", key="md_dl_cif")

        # ---- Downloads + GIF ----
        # .traj bytes for download
        if traj_path and os.path.exists(traj_path):
            with open(traj_path, "rb") as fh:
                st.session_state[_MD_TRAJ] = fh.read()
            st.download_button("⬇️ Download ASE trajectory (.traj)", st.session_state[_MD_TRAJ],
                               file_name="md_trajectory.traj", key="md_dl_traj")

        # optional XYZ
        if xyz_path and os.path.exists(xyz_path):
            with open(xyz_path, "rb") as fh:
                st.session_state[_MD_XYZ] = fh.read()
            st.download_button("⬇️ Download trajectory (XYZ)", st.session_state[_MD_XYZ],
                               file_name="md_trajectory.xyz", key="md_dl_xyz")

        # GIF built from robust .traj
        if make_gif and traj_path and os.path.exists(traj_path):
            frames = ase_read(traj_path, index=":")  # always parses multi-frame properly
            n_frames = len(frames)
            if n_frames < 2:
                st.info("No frames to animate. Increase MD steps or lower the 'Save every Nth step' value.")
            else:
                gif_bytes, n_built = _make_traj_gif_from_frames(frames, forced_symbols=None)
                if n_built >= 2 and gif_bytes:
                    st.session_state[_MD_GIF] = gif_bytes
                    st.image(st.session_state[_MD_GIF],
                             caption=f"Trajectory animation (GIF) — {n_frames} frames",
                             use_container_width=False)
                    st.download_button("⬇️ Download GIF", st.session_state[_MD_GIF],
                                       file_name="md_trajectory.gif", key="md_dl_gif")
                else:
                    st.warning("Animation skipped (encoding failed or memory limit). Try fewer frames or a smaller system.")

    except Exception as exc:
        st.error(f"MD failed: {exc}")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
