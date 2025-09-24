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
# Disable GraphBolt (optional binary) for CPU-only envs
_os.environ.setdefault("DGL_LOAD_GRAPHBOLT", "0")

# ---- stdlib & third-party imports ----
import os, io, uuid, json, warnings, traceback
from typing import Tuple, Optional

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

# MatGL (graceful import with GraphBolt suppression)
try:
    import matgl
    from matgl.apps.pes import Potential
    from matgl.ext.ase import M3GNetCalculator, Relaxer
    _MATGL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # show full detail later in UI
    import importlib
    matgl = None  # type: ignore[assignment]
    Potential = M3GNetCalculator = Relaxer = None  # type: ignore[assignment]
    if isinstance(exc, FileNotFoundError) and "graphbolt" in str(exc).lower():
        try:
            _os.environ["DGL_LOAD_GRAPHBOLT"] = "0"
            matgl = importlib.import_module("matgl")
            from matgl.apps.pes import Potential  # type: ignore
            from matgl.ext.ase import M3GNetCalculator, Relaxer  # type: ignore
            _MATGL_IMPORT_ERROR = None
        except Exception as exc2:
            matgl = None
            Potential = M3GNetCalculator = Relaxer = None
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
            text_data = structure_or_molecule.to(fmt="cif")  # Structure -> CIF
            viewer_format = "cif"
            show_unit_cell = True
        elif isinstance(structure_or_molecule, Molecule):
            text_data = structure_or_molecule.to(fmt="xyz")  # Molecule -> XYZ
            viewer_format = "xyz"
        else:
            # Try ASE conversion then XYZ
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
    Load from MatGL and normalize to a matgl.apps.pes.Potential.
    Returns (potential, error_message).
    """
    if matgl is None:
        return None, f"MatGL is unavailable: {_MATGL_IMPORT_ERROR}"
    try:
        obj = matgl.load_model(name_or_path)  # may return Potential OR a bare model
        if Potential is None:
            return None, "MatGL Potential API unavailable."
        # If it's already a Potential, use it.
        if isinstance(obj, Potential):
            return obj, None
        # Otherwise wrap bare model -> Potential
        pot = Potential(model=obj, calc_forces=True, calc_stresses=True)
        return pot, None
    except Exception as exc:
        return None, f"Failed to load '{name_or_path}': {exc}"

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
            return Molecule.from_str(text, fmt="xyz")
        if lower.endswith((".json", ".mpk", ".mson")):
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
        box = Lattice.cubic(30.0)
        return Structure.from_sites(obj.sites, lattice=box)
    atoms = AseAtomsAdaptor.get_atoms(obj)
    return AseAtomsAdaptor.get_structure(atoms)

def atoms_from(obj):
    """Get ASE Atoms from Structure or Molecule (Molecule boxed as periodic)."""
    if isinstance(obj, Molecule):
        obj = ensure_periodic_structure(obj)
    if isinstance(obj, Structure):
        return AseAtomsAdaptor.get_atoms(obj)
    return AseAtomsAdaptor.get_atoms(obj)

def download_bytes(filename: str, data: bytes, label: str = "Download") -> None:
    st.download_button(label=label, data=data, file_name=filename)

# ---------------------------
# Optional imports: Atomate2 / Jobflow
# ---------------------------
def import_atomate2() -> tuple[Optional[object], Optional[object], Optional[object], Optional[object], Optional[object]]:
    """
    Returns (PhononMaker, ElasticMaker, ForceFieldRelaxMaker, run_locally, SETTINGS)
    or (None,...) if not installed. We show a nice hint in UI if missing.
    """
    try:
        from atomate2.forcefields.flows.phonons import PhononMaker
        from atomate2.forcefields.flows.elastic import ElasticMaker
        from atomate2.forcefields.jobs import ForceFieldRelaxMaker
        from jobflow import run_locally, SETTINGS
        return PhononMaker, ElasticMaker, ForceFieldRelaxMaker, run_locally, SETTINGS
    except Exception as exc:
        return None, None, None, None, None

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="M3GNet Suite — Relaxation • MD • Single-Point • Elastic • Phonons", layout="wide")
st.title("🔬 M3GNet Suite")

# Sidebar: Model selection
with st.sidebar:
    st.header("Model")
    available = try_list_models()
    default_model = available[0] if available else "M3GNet-MP-2018.6.1"
    model_choice = st.selectbox("Choose pretrained PES", [default_model] + [m for m in available if m != default_model])
    model_path_override = st.text_input("...or load from local path / name", value="")
    chosen = model_path_override.strip() or model_choice

    potential, model_err = try_load_model(chosen)
    if model_err:
        st.error(model_err)
        st.caption("Tip: ensure the model name exists or provide a readable local path.")
    else:
        st.success(f"Loaded: {chosen}")

# Input structure
col_up1, col_up2 = st.columns([2, 1], gap="large")
with col_up1:
    st.subheader("Current structure (Input structure)")

    uploaded = st.file_uploader(
        "Upload POSCAR/CONTCAR/CIF/XYZ",
        type=["cif", "POSCAR", "CONTCAR", "xyz", "poscar", "contcar"],
        accept_multiple_files=False,
    )
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
            pmg_obj = parse_uploaded_structure("pasted.cif", text_paste.encode("utf-8"))
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    else:
        # default silicon conventional cell
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
    st.subheader("Quick Notes")
    st.markdown(
        """
        - **Viewer**: Crystals → CIF (with unit cell); molecules → XYZ.
        - **Calculators**: current MatGL `Potential` + `M3GNetCalculator`.
        - **Atomate2** tabs do lazy-import; if missing, you’ll see install hints.
        """
    )
    if _MATGL_IMPORT_ERROR:
        st.warning(f"MatGL import warning: {_MATGL_IMPORT_ERROR}")

st.divider()

# Tabs
tab_about, tab_relax, tab_elastic, tab_phonon, tab_md = st.tabs(
    ["ℹ️ About / Tool", "🔧 Energy Optimization", "🧱 Elastic & Mechanical", "🎼 Phonons (Atomate2)", "🌡️ MD (NVT)"]
)

# ---------------------------
# About tab
# ---------------------------
with tab_about:
    st.subheader("What’s inside")
    st.markdown(
        """
        - **Energy Optimization**: BFGS geometry relaxation on the M3GNet PES.
        - **Elastic & Mechanical** *(Atomate2)*: CHGNet/M3GNet forcefields for
          elastic constants and common stability criteria (e.g., Born).
        - **Phonons** *(Atomate2)*: phonon DOS / band structure, CSV export.
        - **MD (NVT)**: Langevin thermostat; MSD plot + trajectory export.
        """
    )
    st.markdown(
        """
        **Dependencies for Atomate2 tabs** (add to your `requirements.txt` if you
        don’t have them yet):
        ```
        atomate2
        jobflow
        ```
        Optional (for symmetry paths, robust IO): `spglib` (already common), `monty`.
        """
    )

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
        run_relax = st.button("Run Relaxation", use_container_width=True, type="primary", disabled=(potential is None or pmg_obj is None))

    if run_relax and potential is not None and pmg_obj is not None:
        try:
            atoms = atoms_from(pmg_obj)
            atoms.calc = M3GNetCalculator(potential=potential)

            traj_positions = []
            def _record(a=atoms):
                traj_positions.append(a.get_positions().copy())

            dyn = BFGS(atoms, logfile=None, maxstep=0.2)
            dyn.attach(_record, interval=1)
            dyn.run(fmax=float(fmax), steps=int(max_steps))

            e_final = atoms.get_potential_energy()
            f_max = np.abs(atoms.get_forces()).max()

            st.success(f"Optimization finished. E = {e_final:.6f} eV | max|F| = {f_max:.4f} eV/Å")

            # Export final structure
            final_struct = AseAtomsAdaptor.get_structure(atoms)
            cif_bytes = final_struct.to(fmt="cif").encode("utf-8")
            download_bytes("relaxed_structure.cif", cif_bytes, label="⬇️ Download relaxed CIF")

            if traj_want and traj_positions:
                buf = io.StringIO()
                for pos in traj_positions:
                    tmp = atoms.copy()
                    tmp.set_positions(pos)
                    ase_write(buf, tmp, format="xyz")
                download_bytes("relaxation_trajectory.xyz", buf.getvalue().encode("utf-8"), label="⬇️ Download relaxation trajectory (XYZ)")

        except Exception as exc:
            st.error("Relaxation failed. See details in the expandable traceback below.")
            with st.expander("Traceback"):
                st.code("".join(traceback.format_exception(exc)))

# ---------------------------
# Elastic constants tab (Atomate2)
# ---------------------------
with tab_elastic:
    st.subheader("Elastic Constants & Mechanical Stability (Atomate2 ForceFields)")
    PhononMaker, ElasticMaker, ForceFieldRelaxMaker, run_locally, SETTINGS = import_atomate2()

    if ElasticMaker is None:
        st.error(
            "Atomate2 / Jobflow not available. Add to requirements:\n"
            "`atomate2` and `jobflow`"
        )
    else:
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            ff_name = st.selectbox("Force Field", ["M3GNet", "CHGNet"], index=0)
        with col_e2:
            fmax_el = st.number_input("Relax fmax (eV/Å)", value=1e-4, min_value=1e-6, max_value=1e-2, step=1e-4, format="%.6f")
        with col_e3:
            run_elastic = st.button("Run Elastic Workflow", type="primary", disabled=(pmg_obj is None))

        if run_elastic and pmg_obj is not None:
            try:
                elastic_flow = ElasticMaker(
                    bulk_relax_maker=ForceFieldRelaxMaker(
                        force_field_name=ff_name,
                        relax_cell=True,
                        relax_kwargs={"fmax": float(fmax_el)}
                    ),
                    elastic_relax_maker=ForceFieldRelaxMaker(
                        force_field_name=ff_name,
                        relax_cell=False,
                        relax_kwargs={"fmax": float(fmax_el)}
                    ),
                ).make(structure=pmg_obj)

                st.info("Launching elastic workflow locally (jobflow).")
                responses = run_locally(elastic_flow, create_folders=True)

                store = SETTINGS.JOB_STORE
                store.connect()
                result = store.query_one(
                    {"name": "fit_elastic_tensor"},
                    properties=[
                        "output.elastic_tensor",
                        "output.derived_properties",
                    ],
                    load=True,
                    sort={"completed_at": -1}
                )
                if result is None:
                    st.warning("No elastic tensor found in store yet.")
                else:
                    et = result["output"]["elastic_tensor"]
                    dp = result["output"]["derived_properties"]
                    st.success("Elastic tensor (IEEE) and derived properties")
                    st.code(json.dumps(et.get("ieee_format", et), indent=2))
                    st.code(json.dumps(dp, indent=2))

            except Exception as exc:
                st.error("Elastic workflow failed. See details below.")
                with st.expander("Traceback"):
                    st.code("".join(traceback.format_exception(exc)))

# ---------------------------
# Phonons tab (Atomate2)
# ---------------------------
with tab_phonon:
    st.subheader("Phonons: DOS & Band Structure (Atomate2 ForceFields)")
    PhononMaker, ElasticMaker, ForceFieldRelaxMaker, run_locally, SETTINGS = import_atomate2()

    if PhononMaker is None:
        st.error(
            "Atomate2 / Jobflow not available. Add to requirements:\n"
            "`atomate2` and `jobflow`"
        )
    else:
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            min_length = st.number_input("Supercell min length (Å)", value=15.0, min_value=8.0, max_value=30.0, step=0.5)
        with col_p2:
            store_fc = st.checkbox("Store force constants", value=False)
        with col_p3:
            run_ph = st.button("Run Phonon Workflow", type="primary", disabled=(pmg_obj is None))

        if run_ph and pmg_obj is not None:
            try:
                phonon_flow = PhononMaker(
                    min_length=float(min_length),
                    store_force_constants=bool(store_fc),
                ).make(structure=pmg_obj)

                st.info("Launching phonon workflow locally (jobflow).")
                _ = run_locally(phonon_flow, create_folders=True)

                # Query results
                store = SETTINGS.JOB_STORE
                store.connect()
                result = store.query_one(
                    {"name": "generate_frequencies_eigenvectors"},
                    properties=[
                        "output.phonon_dos",
                        "output.phonon_bandstructure",
                    ],
                    load=True,
                    sort={"completed_at": -1}
                )

                if result is None:
                    st.warning("No phonon results found yet.")
                else:
                    # Build Phonon objects
                    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
                    from pymatgen.phonon.dos import PhononDos
                    from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter

                    ph_bs = PhononBandStructureSymmLine.from_dict(result['output']['phonon_bandstructure'])
                    ph_dos = PhononDos.from_dict(result['output']['phonon_dos'])

                    # Plot DOS
                    dos_plot = PhononDosPlotter()
                    dos_plot.add_dos(label='Phonon DOS', dos=ph_dos)
                    ax_dos = dos_plot.get_plot()
                    fig_dos = ax_dos.get_figure()
                    buf_dos = io.BytesIO()
                    fig_dos.savefig(buf_dos, dpi=300, bbox_inches="tight")
                    plt.close(fig_dos)
                    buf_dos.seek(0)
                    st.image(buf_dos.read(), caption="Phonon DOS", use_container_width=True)

                    # Plot Band Structure
                    bs_plot = PhononBSPlotter(bs=ph_bs)
                    ax_bs = bs_plot.get_plot()
                    fig_bs = ax_bs.get_figure()
                    buf_bs = io.BytesIO()
                    fig_bs.savefig(buf_bs, dpi=300, bbox_inches="tight")
                    plt.close(fig_bs)
                    buf_bs.seek(0)
                    st.image(buf_bs.read(), caption="Phonon Band Structure", use_container_width=True)

                    # CSV exports
                    # DOS
                    import csv
                    dos_csv = io.StringIO()
                    frequencies = ph_dos.frequencies  # THz
                    total_dos = ph_dos.densities
                    writer = csv.writer(dos_csv)
                    writer.writerow(["Frequency (THz)", "Total DOS"])
                    for fval, dval in zip(frequencies, total_dos):
                        writer.writerow([fval, dval])
                    download_bytes("phonon_dos.csv", dos_csv.getvalue().encode("utf-8"), label="⬇️ Download Phonon DOS (CSV)")

                    # Bands
                    bands = ph_bs.bands  # shape (n_branches, n_points)
                    distances = ph_bs.distance  # length n_points
                    bands_T = bands.T           # (n_points, n_branches)
                    bs_csv = io.StringIO()
                    writer = csv.writer(bs_csv)
                    header = ["Distance (1/Å)"] + [f"Branch {i+1} (THz)" for i in range(bands.shape[0])]
                    writer.writerow(header)
                    for i, d in enumerate(distances):
                        row = [d] + list(bands_T[i])
                        writer.writerow(row)
                    writer.writerow([])
                    writer.writerow(["High Symmetry Points"])
                    for label, dist in ph_bs.labels_dict.items():
                        writer.writerow([label, dist])
                    download_bytes("phonon_bandstructure.csv", bs_csv.getvalue().encode("utf-8"), label="⬇️ Download Phonon Bands (CSV)")

                    st.success("Phonon analyses complete and files exported.")

            except Exception as exc:
                st.error("Phonon workflow failed. See details below.")
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
        run_md = st.button("Run MD", use_container_width=True, type="primary", disabled=(potential is None or pmg_obj is None))

    if run_md and potential is not None and pmg_obj is not None:
        try:
            atoms = atoms_from(pmg_obj)
            atoms.calc = M3GNetCalculator(potential=potential)

            # Initialize velocities
            MaxwellBoltzmannDistribution(atoms, temperature_K=float(T))

            # Langevin thermostat: friction in 1/ps -> convert to ASE units (1/fs)
            gamma_fs = float(friction) / 1000.0  # 1/ps -> 1/fs
            dyn = Langevin(atoms, timestep=float(dt_fs) * units.fs, temperature_K=float(T), friction=gamma_fs)

            positions = []
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

# Footer note
st.caption("Note: Crystals visualize as CIF; molecules visualize as XYZ. ASE calculator uses M3GNetCalculator(potential=Potential).")
