# tabs/md_tab.py
from __future__ import annotations
import io
import os
import tempfile
import traceback
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
from pymatgen.io.ase import AseAtomsAdaptor

from core.struct import atoms_from
from core.model import get_calculator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def _online_msd_state():
    # store reference positions and running MSD
    return {"r0": None, "msd": []}


def _online_msd_update(state, r: np.ndarray):
    if state["r0"] is None:
        state["r0"] = r.copy()
        state["msd"].append(0.0)
        return
    dr = r - state["r0"]
    # mean over atoms of squared displacement
    state["msd"].append(float((dr * dr).sum(axis=1).mean()))


def _plot_msd(msd_vals, dt_fs: float):
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


def _make_traj_gif(xyz_path: str, forced_symbols: str | None = None) -> bytes:
    """
    Render a GIF animation from an XYZ trajectory. If forced_symbols is provided
    (e.g., '64Si'), all frames' chemical symbols are set accordingly.
    """
    try:
        atoms_list = ase_read(xyz_path, index=":")
        if forced_symbols:
            for a in atoms_list:
                a.set_chemical_symbols(forced_symbols)

        fig, ax = plt.subplots(figsize=(4, 4))

        def update(i):
            ax.clear()
            from ase.visualize.plot import plot_atoms
            plot_atoms(atoms_list[i], ax, radii=0.8, rotation=("45x,45y,0z"))
            ax.set_title(f"Frame {i + 1}")
            ax.axis("off")

        ani = FuncAnimation(fig, update, frames=len(atoms_list), interval=200)  # 200 ms per frame
        buf = io.BytesIO()
        ani.save(buf, writer="pillow", dpi=200, fps=60)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return b""


def md_tab(pmg_obj, model_family: str, variant: str):
    st.subheader("Molecular Dynamics")

    # Top controls
    c0, c00 = st.columns([1, 1])
    with c0:
        ensemble = st.selectbox(
            "Ensemble",
            ["NVE", "NVT (Langevin)", "NPT (Berendsen)"],
            key="md_ensemble",
        )
    with c00:
        if ensemble.startswith("NPT") and NPTBerendsen is None:
            st.warning("NPT Berendsen integrator unavailable in this environment; choose NVE or NVT.", icon="⚠️")

    # Temperature & timestep
    cT1, cT2, cT3 = st.columns(3)
    with cT1:
        T_init = st.number_input(
            "Initial temperature (K)",
            value=300, min_value=1, max_value=5000, step=10, key="md_T_init"
        )
    with cT2:
        T_target = st.number_input(
            "Target temperature (K)",
            value=300, min_value=1, max_value=5000, step=10, key="md_T_target"
        )
    with cT3:
        dt_fs = st.number_input(
            "Timestep (fs)",
            value=1.0, min_value=0.1, max_value=10.0, step=0.1, format="%.1f", key="md_dt"
        )

    # Steps, friction, pressure
    c1, c2, c3 = st.columns(3)
    with c1:
        steps = st.number_input(
            "MD steps",
            value=2000, min_value=10, max_value=200000, step=100, key="md_steps"
        )
    with c2:
        friction = st.number_input(
            "Friction γ (1/ps) [NVT]",
            value=1.0, min_value=0.01, max_value=100.0, step=0.1, format="%.2f", key="md_gamma"
        )
    with c3:
        pressure_bar = st.number_input(
            "Pressure (bar) [NPT]",
            value=1.0, min_value=0.0, max_value=100.0, step=0.5, key="md_pbar"
        )

    # Output options
    c4, c5, c6 = st.columns(3)
    with c4:
        save_xyz = st.checkbox("Stream trajectory to XYZ", value=False, key="md_savexyz")
    with c5:
        stride = st.number_input("Save every Nth step", value=25, min_value=1, max_value=1000, step=1, key="md_stride")
    with c6:
        make_gif = st.checkbox("Render GIF animation", value=False, key="md_makegif")

    if pmg_obj is None:
        st.info("Upload a structure to run MD.")
        return

    if not st.button("Run MD", type="primary", key="md_run"):
        return

    try:
        # reset stop flag before we start
        st.session_state.stop_requested = False

        # Build ASE Atoms + calculator
        atoms = atoms_from(pmg_obj)
        atoms.calc = get_calculator(model_family, variant)

        # Initialize velocities at initial temperature
        MaxwellBoltzmannDistribution(atoms, temperature_K=float(T_init))

        # Choose integrator
        dt = float(dt_fs) * units.fs
        if ensemble.startswith("NVE"):
            dyn = VelocityVerlet(atoms, dt)
        elif ensemble.startswith("NVT"):
            gamma_fs = float(friction) / 1000.0  # 1/ps → 1/fs
            dyn = Langevin(atoms, timestep=dt, temperature_K=float(T_target), friction=gamma_fs)
        else:
            if NPTBerendsen is None:
                st.error("NPT not available. Choose NVE or NVT.")
                return
            p_target = float(pressure_bar) * 1e5  # bar → Pa
            dyn = NPTBerendsen(
                atoms,
                timestep=dt,
                temperature_K=float(T_target),
                taut=100.0 * units.fs,
                pressure=p_target,
                compressibility=4.5e-10,  # typical for solids
            )

        # Progress UI
        progress = st.progress(0, text="Starting MD…")
        pct_label = st.empty()

        # Online MSD
        msd_state = _online_msd_state()

        # Stream trajectory to a temp file if requested
        tmp_path, xyz_file = (None, None)
        if save_xyz:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
            tmp_path = tmp.name
            tmp.close()
            st.session_state.tmp_paths.append(tmp_path)
            xyz_file = open(tmp_path, "w")

        # Per-step callback
        step_idx = {"i": 0}

        def on_step():
            if st.session_state.stop_requested:
                raise RuntimeError("Stop requested by user.")
            step_idx["i"] += 1
            i = step_idx["i"]

            pos = atoms.get_positions()
            _online_msd_update(msd_state, pos)

            if save_xyz and (i % int(st.session_state.get("md_stride", 25)) == 0):
                s = io.StringIO()
                frame = atoms.copy()
                frame.set_positions(pos)
                ase_write(s, frame, format="xyz")
                xyz_file.write(s.getvalue())

            pct = min(int(100 * i / steps), 99)
            progress.progress(pct, text=f"MD running… {pct}%")
            pct_label.write(f"**Progress:** {pct}%")

        dyn.attach(on_step, interval=1)
        dyn.run(int(steps))

        if xyz_file is not None:
            xyz_file.flush()
            xyz_file.close()

        progress.progress(100, text="Done")
        pct_label.write("**Progress:** 100%")

        # MSD plot
        png = _plot_msd(msd_state["msd"], float(dt_fs))
        if png:
            st.image(png, caption="MSD vs time", use_container_width=False)

        # Last frame (CIF)
        final_struct = AseAtomsAdaptor.get_structure(atoms)
        st.download_button(
            "⬇️ Download last frame (CIF)",
            final_struct.to(fmt="cif").encode(),
            file_name="md_last_frame.cif",
            key="md_dl_cif",
        )

        # Trajectory (XYZ) + optional GIF
        if tmp_path:
            with open(tmp_path, "rb") as fh:
                xyz_bytes = fh.read()
            st.download_button(
                "⬇️ Download trajectory (XYZ, streamed)",
                xyz_bytes,
                file_name="md_trajectory.xyz",
                key="md_dl_xyz",
            )
            if make_gif:
                gif_bytes = _make_traj_gif(tmp_path, forced_symbols=None)  # set to e.g. "64Si" if you want
                if gif_bytes:
                    st.image(gif_bytes, caption="Trajectory animation (GIF)", use_container_width=False)
                    st.download_button("⬇️ Download GIF", gif_bytes, file_name="md_trajectory.gif", key="md_dl_gif")

    except Exception as exc:
        st.error(f"MD failed: {exc}")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
