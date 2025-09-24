from __future__ import annotations

# ---- single-threaded runtime (set BEFORE imports that spin threads) ----
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_os.environ.setdefault("TORCH_NUM_THREADS", "1")
# Some DGL CPU wheels ship without optional GraphBolt; disable proactively.
_os.environ.setdefault("DGL_LOAD_GRAPHBOLT", "0")

# ---- stdlib & third-party imports ----
import os, io, uuid, json, time, warnings, traceback
from typing import Tuple

import numpy as np
import streamlit as st

from pymatgen.core import Structure, Molecule
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor

import ase
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase.io import write as ase_write

# MatGL (graceful import + GraphBolt suppression on retry if needed)
try:
    import matgl
    from matgl.ext.ase import PESCalculator, Relaxer
    _MATGL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # show full detail later in UI
    import importlib
    matgl = None  # type: ignore[assignment]
    PESCalculator = Relaxer = None  # type: ignore[assignment]
    if isinstance(exc, FileNotFoundError) and "graphbolt" in str(exc).lower():
        try:
            _os.environ["DGL_LOAD_GRAPHBOLT"] = "0"
            matgl = importlib.import_module("matgl")
            from matgl.ext.ase import PESCalculator, Relaxer  # type: ignore
            _MATGL_IMPORT_ERROR = None
        except Exception as exc2:
            matgl = None
            PESCalculator = Relaxer = None
            _MATGL_IMPORT_ERROR = exc2
    else:
        _MATGL_IMPORT_ERROR = exc

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

# ---------------------------
# Quick sanity guard for ASE
# ---------------------------
from packaging import version as _v
if _v.parse(ase.__version__) < _v.parse("3.23.0"):
    raise RuntimeError(
        f"Your ASE ({ase.__version__}) is too old. Please upgrade: pip install --upgrade ase (need >= 3.23.0)"
    )

# ---------------------------
# Helpers
# ---------------------------
def lattice_caption(s: Structure) -> str:
    a, b, c = s.lattice.abc
    alpha, beta, gamma = s.lattice.angles
    return (
        f"a,b,c = {a:.3f}, {b:.3f}, {c:.3f} Å | "
        f"α,β,γ = {alpha:.2f}, {beta:.2f}, {gamma:.2f}° | "
        f"natoms={len(s)}"
    )

def msd_and_plot(positions: np.ndarray, dt_fs: float) -> Tuple[np.ndarray, np.ndarray, bytes]:
    """
    positions: shape (n_steps, n_atoms, 3) in Å
    dt_fs: time step in femtoseconds
    """
    r0 = positions[0]
    dr = positions - r0
    msd = (dr**2).sum(axis=2).mean(axis=1)
    t_ps = np.arange(len(msd)) * dt_fs / 1000.0

    fig = plt.figure()
    plt.plot(t_ps, msd)
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD (Å$^2$)")
    plt.title("Mean Squared Displacement")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return msd, t_ps, buf.read()

def render_structure_viewer(structure_or_molecule, height: int = 480) -> None:
    """
    3D viewer using 3Dmol.js embedded in Streamlit.

    - pymatgen.Structure -> CIF (3Dmol format 'cif'), with unit cell shown
    - pymatgen.Molecule  -> XYZ (3Dmol format 'xyz')
    - Fallback: try ASE to XYZ
    """
    viewer_format = None
    text_data = None
    show_unit_cell = False

    try:
        if isinstance(structure_or_molecule, Structure):
            # Structure does NOT support 'xyz' in Structure.to(); use CIF.
            text_data = structure_or_molecule.to(fmt="cif")
            viewer_format = "cif"
            show_unit_cell = True
        elif isinstance(structure_or_molecule, Molecule):
            # Molecule supports 'xyz'
            text_data = structure_or_molecule.to(fmt="xyz")
            viewer_format = "xyz"
        else:
            # Try ASE conversion, then XYZ
            atoms = AseAtomsAdaptor.get_atoms(structure_or_molecule)
            buf = io.StringIO()
            ase_write(buf, atoms, format="xyz")
            text_data = buf.getvalue()
            viewer_format = "xyz"
    except Exception as exc:
        st.error(f"Failed to serialize structure for viewer: {exc}")
        return

    container_id = f"structure-viewer-{uuid.uuid4().hex}"
    data_json = json.dumps(text_data)

    html = f"""
    <div id="{container_id}" style="height: {height}px; position: relative;"></div>
    <script src="https://3dmol.csb.pitt.edu/build/3Dmol.js"></script>
    <script>
      (function() {{
        var viewer = $3Dmol.createViewer('{container_id}', {{backgroundColor: 'white'}});
        var data = {data_json};
        viewer.addModel(data, '{viewer_format}');
        viewer.setStyle({{}}, {{stick: {{radius: 0.18}}, sphere: {{scale: 0.25}}}});
        {"viewer.addUnitCell();" if show_unit_cell else ""}
        viewer.zoomTo();
        viewer.render();
      }})();
    </script>
    """
    st.components.v1.html(html, height=height)

def try_list_models() -> list[str]:
    if matgl is None:
        return []
    try:
        return list(matgl.get_available_pretrained_models())
    except Exception:
        return []

def try_load_model(name_or_path: str):
    """
    Load a pretrained M3GNet PES by name OR local path.
    IMPORTANT: Do NOT pass device= to matgl.load_model (older matgl forwards kwargs).
    After load, move to CPU if .to exists.
    Returns (model, error_message) where error_message is None on success.
    """
    if matgl is None:
        return None, f"MatGL is unavailable: {_MATGL_IMPORT_ERROR}"
    try:
        model = matgl.load_model(name_or_path)  # <-- no device kwarg here
        try:
            model.to("cpu")  # some matgl models expose .to()
        except Exception:
            pass
        return model, None
    except Exception as exc:
        return None, f"Failed to load model '{name_or_path}': {exc}"

def parse_uploaded_structure(name: str, content: bytes):
    """Return either Structure or Molecule based on file content."""
    text = content.decode("utf-8", errors="ignore")
    lower = name.lower()

    try:
        if lower.endswith((".cif",)):
            return Structure.from_str(text, fmt="cif")
        if lower in ("poscar", "contcar") or "poscar" in lower or "contcar" in lower:
            return Structure.from_str(text, fmt="poscar")
        if lower.endswith((".xyz",)):
            # XYZ => Molecule (non-periodic)
            return Molecule.from_str(text, fmt="xyz")
        if lower.endswith((".json", ".mpk", ".mson")):
            # Pymatgen as_dict formats (Structure or Molecule)
            obj = Structure.from_str(text, fmt="json")
            return obj
    except Exception:
        pass  # fall through to generic tries

    # Try Structure first, then Molecule
    for fmt in ("cif", "poscar"):
        try:
            return Structure.from_str(text, fmt=fmt)
        except Exception:
            continue
    try:
        return Molecule.from_str(text, fmt="xyz")
    except Exception as exc:
        raise ValueError(f"Could not parse structure/molecule from '{name}': {exc}")

def ensure_periodic_structure(obj):
    """Convert Molecule to a dummy periodic Structure (large cubic box) if needed."""
    if isinstance(obj, Structure):
        return obj
    if isinstance(obj, Molecule):
        # Put molecule in a large cubic box (periodic) so ASE/MatGL can run
        box = Lattice.cubic(30.0)
        return Structure.from_sites(obj.sites, lattice=box)
    # try ASE -> Structure
    atoms = AseAtomsAdaptor.get_atoms(obj)
    return AseAtomsAdaptor.get_structure(atoms)

def atoms_from(obj):
    """Get ASE Atoms from Structure or Molecule (Molecule boxed as periodic)."""
    if isinstance(obj, Molecule):
        obj = ensure_periodic_structure(obj)
    if isinstance(obj, Structure):
        return AseAtomsAdaptor.get_atoms(obj)
    # Try best-effort
    return AseAtomsAdaptor.get_atoms(obj)

def download_bytes(filename: str, data: bytes, label: str = "Download") -> None:
    st.download_button(label=label, data=data, file_name=filename)

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="M3GNet Suite — Relaxation • MD • Single-Point", layout="wide")
st.title("🔬 M3GNet Suite — Relaxation • MD • Single-Point")

# Sidebar: Model selection
with st.sidebar:
    st.header("Model")
    available = try_list_models()
    default_model = available[0] if available else "M3GNet-MP-2018.6.1"
    model_choice = st.selectbox("Choose pretrained PES", [default_model] + [m for m in available if m != default_model])
    model_path_override = st.text_input("...or load from local path / name", value="")
    chosen = model_path_override.strip() or model_choice

    model, model_err = try_load_model(chosen)
    if model_err:
        st.error(model_err)
        st.caption("Tip: ensure the model name exists or provide a readable local path.")
    else:
        st.success(f"Loaded: {chosen}")

# Input structure
col_up1, col_up2 = st.columns([2, 1], gap="large")
with col_up1:
    st.subheader("Current structure (Input structure)")

    uploaded = st.file_uploader("Upload POSCAR/CONTCAR/CIF/XYZ", type=["cif", "POSCAR", "CONTCAR", "xyz", "poscar", "contcar"], accept_multiple_files=False)
    text_paste = st.text_area("...or paste structure text (POSCAR/CIF/XYZ)", height=180)

    pmg_obj = None
    parse_error = None
    if uploaded is not None:
        try:
            pmg_obj = parse_uploaded_structure(uploaded.name, uploaded.read())
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    elif text_paste.strip():
        try:
            # Name heuristic for parsing hint
            hint = "POSCAR" if text_paste.strip().splitlines()[0].strip() and len(text_paste.split()) < 30 else "cif"
            pmg_obj = parse_uploaded_structure(f"pasted.{hint}", text_paste.encode("utf-8"))
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    else:
        # Provide a tiny default structure (Si)
        lat = Lattice.cubic(5.431)
        pmg_obj = Structure(lat, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
        st.info("No file pasted/uploaded. Using default diamond Si conventional cell.")

    if parse_error:
        st.error(parse_error)

    # Show viewer + caption
    if pmg_obj is not None:
        if isinstance(pmg_obj, Structure):
            st.caption(lattice_caption(pmg_obj))
        else:
            st.caption(f"Molecule with {len(pmg_obj)} atoms")
        render_structure_viewer(pmg_obj, height=480)

with col_up2:
    st.subheader("About")
    st.markdown(
        """
        - **Viewer fix**: Crystals are exported as **CIF** (not XYZ) to 3Dmol, avoiding the `Structure.to(fmt="xyz")` error.
        - **Molecules**: rendered as **XYZ**.  
        - **Compute** uses MatGL's **PESCalculator/Relaxer** via ASE.
        """
    )
    if _MATGL_IMPORT_ERROR:
        st.warning(f"MatGL import warning: {_MATGL_IMPORT_ERROR}")

st.divider()

# Tabs for workflows
tab_relax, tab_md, tab_sp = st.tabs(["🔧 Relaxation", "🌡️ MD (NVT, Langevin)", "⚡ Single-point"])

# ---------------------------
# Relaxation Tab
# ---------------------------
with tab_relax:
    st.subheader("Geometry Optimization (BFGS on M3GNet PES)")
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fmax = st.number_input("Force convergence (eV/Å)", value=0.05, min_value=0.001, max_value=1.0, step=0.01, format="%.3f")
        max_steps = st.number_input("Max optimization steps", value=200, min_value=10, max_value=5000, step=10)
    with col_r2:
        traj_want = st.checkbox("Record trajectory (XYZ)", value=True)
        run_relax = st.button("Run Relaxation", use_container_width=True, type="primary", disabled=(model is None or pmg_obj is None))

    if run_relax and model is not None and pmg_obj is not None:
        try:
            atoms = atoms_from(pmg_obj)
            atoms.calc = PESCalculator(model=model)

            traj_positions = []
            traj_symbols = [a.symbol for a in atoms]

            def _record(a=atoms):
                traj_positions.append(a.get_positions().copy())

            dyn = BFGS(atoms, logfile=None, maxstep=0.2)
            dyn.attach(_record, interval=1)
            dyn.run(fmax=fmax, steps=int(max_steps))

            e_final = atoms.get_potential_energy()
            f_max = np.abs(atoms.get_forces()).max()

            st.success(f"Optimization finished. E = {e_final:.6f} eV | max|F| = {f_max:.4f} eV/Å")

            # Export final structure
            final_struct = AseAtomsAdaptor.get_structure(atoms)
            cif_bytes = final_struct.to(fmt="cif").encode("utf-8")
            download_bytes("relaxed_structure.cif", cif_bytes, label="⬇️ Download relaxed CIF")

            if traj_want and traj_positions:
                # Build XYZ trajectory text
                frames = []
                for pos in traj_positions:
                    buf = io.StringIO()
                    # write a single frame
                    tmp = atoms.copy()
                    tmp.set_positions(pos)
                    ase_write(buf, tmp, format="xyz")
                    frames.append(buf.getvalue())
                xyz_traj = "\n".join(frames).encode("utf-8")
                download_bytes("relaxation_trajectory.xyz", xyz_traj, label="⬇️ Download relaxation trajectory (XYZ)")

        except Exception as exc:
            st.error("Relaxation failed. See details in the expandable traceback below.")
            with st.expander("Traceback"):
                st.code("".join(traceback.format_exception(exc)))

# ---------------------------
# MD Tab
# ---------------------------
with tab_md:
    st.subheader("Molecular Dynamics (NVT via Langevin) on M3GNet PES")
    colm1, colm2, colm3 = st.columns(3)
    with colm1:
        T = st.number_input("Temperature (K)", value=300, min_value=1, max_value=3000, step=10)
        dt_fs = st.number_input("Timestep (fs)", value=1.0, min_value=0.1, max_value=10.0, step=0.1, format="%.1f")
    with colm2:
        steps = st.number_input("MD steps", value=2000, min_value=10, max_value=200000, step=100)
        friction = st.number_input("Friction γ (1/ps)", value=1.0, min_value=0.01, max_value=100.0, step=0.1, format="%.2f")
    with colm3:
        save_xyz = st.checkbox("Save trajectory (XYZ)", value=True)
        run_md = st.button("Run MD", use_container_width=True, type="primary", disabled=(model is None or pmg_obj is None))

    if run_md and model is not None and pmg_obj is not None:
        try:
            atoms = atoms_from(pmg_obj)
            atoms.calc = PESCalculator(model=model)

            # Initialize velocities
            MaxwellBoltzmannDistribution(atoms, temperature_K=float(T))

            # Langevin thermostat: friction in 1/ps -> convert to ASE units (1/fs)
            gamma_fs = float(friction) / 1000.0  # 1/ps -> 1/fs
            dyn = Langevin(atoms, timestep=float(dt_fs) * units.fs, temperature_K=float(T), friction=gamma_fs)

            positions = []
            symbols = [a.symbol for a in atoms]

            def _snapshot():
                positions.append(atoms.get_positions().copy())

            dyn.attach(_snapshot, interval=1)
            dyn.run(int(steps))

            positions = np.array(positions)  # (n_steps, n_atoms, 3)
            msd, t_ps, png = msd_and_plot(positions, float(dt_fs))
            st.image(png, caption="MSD vs Time", use_container_width=True)

            # Export last frame + full trajectory if desired
            if save_xyz:
                buf = io.StringIO()
                for pos in positions:
                    tmp = atoms.copy()
                    tmp.set_positions(pos)
                    ase_write(buf, tmp, format="xyz")
                download_bytes("md_trajectory.xyz", buf.getvalue().encode("utf-8"), label="⬇️ Download MD trajectory (XYZ)")

            # Export final structure
            atoms.set_positions(positions[-1])
            final_struct = AseAtomsAdaptor.get_structure(atoms)
            cif_bytes = final_struct.to(fmt="cif").encode("utf-8")
            download_bytes("md_last_frame.cif", cif_bytes, label="⬇️ Download last frame (CIF)")

        except Exception as exc:
            st.error("MD failed. See details in the expandable traceback below.")
            with st.expander("Traceback"):
                st.code("".join(traceback.format_exception(exc)))

# ---------------------------
# Single-point Tab
# ---------------------------
with tab_sp:
    st.subheader("Single-point Energy / Forces")
    run_sp = st.button("Run Single-point", use_container_width=True, type="primary", disabled=(model is None or pmg_obj is None))

    if run_sp and model is not None and pmg_obj is not None:
        try:
            atoms = atoms_from(pmg_obj)
            atoms.calc = PESCalculator(model=model)
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            st.success(f"E = {e:.6f} eV | max|F| = {np.abs(f).max():.4f} eV/Å")

            # Save forces as CSV
            import pandas as pd
            df = pd.DataFrame(f, columns=["Fx (eV/Å)", "Fy (eV/Å)", "Fz (eV/Å)"])
            csv = df.to_csv(index=False).encode("utf-8")
            download_bytes("single_point_forces.csv", csv, label="⬇️ Download forces (CSV)")

        except Exception as exc:
            st.error("Single-point failed. See details in the expandable traceback below.")
            with st.expander("Traceback"):
                st.code("".join(traceback.format_exception(exc)))

# Footer note
st.caption("Note: For crystals, viewer uses CIF (not XYZ). Molecules render as XYZ. This avoids pymatgen Structure.to(fmt='xyz') errors and works smoothly with 3Dmol.")
