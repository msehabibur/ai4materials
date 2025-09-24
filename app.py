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

import os, time, io, warnings, traceback, json, uuid
from typing import List, Tuple
import numpy as np
import streamlit as st

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor

import ase
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin

import matgl
from matgl.ext.ase import PESCalculator, Relaxer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

# ---------------------------
# Version guard for ASE
# ---------------------------
from packaging import version as _v
if _v.parse(ase.__version__) < _v.parse("3.23.0"):
    raise RuntimeError(
        f"Your ASE ({ase.__version__}) is too old. Please upgrade: pip install --upgrade ase (need >= 3.23.0)"
    )

# ---------------------------
# Helpers
# ---------------------------
def save_temp(file_bytes: bytes, name: str, workdir: str) -> str:
    path = os.path.join(workdir, name)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path

def lattice_caption(s: Structure) -> str:
    a, b, c = s.lattice.abc
    alpha, beta, gamma = s.lattice.angles
    return (f"a,b,c = {a:.3f}, {b:.3f}, {c:.3f} Å | "
            f"α,β,γ = {alpha:.2f}, {beta:.2f}, {gamma:.2f}° | "
            f"natoms={len(s)}")

def msd_and_plot(positions: np.ndarray, dt_fs: float) -> Tuple[np.ndarray, np.ndarray, bytes]:
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


def render_structure_viewer(structure: Structure, height: int = 480) -> None:
    """Render a 3D structure viewer using 3Dmol.js embedded in Streamlit."""

    xyz = structure.to(fmt="xyz")
    xyz_json = json.dumps(xyz)
    container_id = f"structure-viewer-{uuid.uuid4().hex}"

    html = f"""
    <div id="{container_id}" style="height: {height}px; position: relative;"></div>
    <script src="https://3dmol.csb.pitt.edu/build/3Dmol.js"></script>
    <script>
    (function() {{
        var viewer = $3Dmol.createViewer('{container_id}', {{backgroundColor: 'white'}});
        var xyz = {xyz_json};
        viewer.addModel(xyz, 'xyz');
        viewer.setStyle({{}}, {{stick: {{radius: 0.18}}, sphere: {{scale: 0.25}}}});
        viewer.addUnitCell();
        viewer.zoomTo();
        viewer.render();
    }})();
    </script>
    """

    st.components.v1.html(html, height=height)

def try_list_models() -> list[str]:
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
    try:
        model = matgl.load_model(name_or_path)  # <-- no device kwarg here
        # Move to CPU if supported
        try:
            model.to("cpu")  # some matgl models expose .to()
        except Exception:
            pass
        return model, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"{e}\n\nTraceback:\n{tb}"

# ---------------------------
# UI Setup
# ---------------------------
st.set_page_config(page_title="M3GNet — Relax • MD • Single Point", layout="wide")
st.title("🔬 M3GNet Suite — Relaxation • MD • Single-Point")

with st.sidebar:
    st.header("Data & Viewer")
    uploaded = st.file_uploader("Upload CIF (or leave empty for demo CsCl)", type=["cif"])
    workdir = st.text_input("Workdir", value=os.path.join(os.getcwd(), "tmp_ml"))
    os.makedirs(workdir, exist_ok=True)

    st.subheader("Model selection / loading")
    avail = try_list_models()
    default_name = "M3GNet-MP-2021.2.8-PES"
    if avail:
        model_name = st.selectbox("Pick a pretrained model", options=avail, index=avail.index(default_name) if default_name in avail else 0)
    else:
        st.caption("Could not list pretrained models (no internet or registry blocked). Type a name or local path:")
        model_name = st.text_input("Model name or local path", value=default_name)

    local_override = st.text_input("OR load from local path (folder/file)", value="", help="If provided, this overrides the selected registry model.")
    load_button = st.button("🔁 Load / Reload model (CPU)")

# ---- model loading (robust) ----
if "m3gnet_err" not in st.session_state:
    st.session_state.m3gnet_err = None
if "m3gnet_pot" not in st.session_state or load_button:
    chosen = (local_override or model_name).strip()
    pot, err = try_load_model(chosen)
    st.session_state.m3gnet_pot = pot
    st.session_state.m3gnet_err = err

if st.session_state.m3gnet_err:
    st.error("Failed to load model. See details below.")
    st.code(st.session_state.m3gnet_err)
    st.stop()

pot = st.session_state.m3gnet_pot
if pot is None:
    st.warning("No model loaded yet. Choose a model name or local path in the sidebar, then click 'Load / Reload model'.")
    st.stop()

# ---------------------------
# Load structure
# ---------------------------
if uploaded is not None:
    cif_path = save_temp(uploaded.getbuffer(), f"{int(time.time())}_{uploaded.name}", workdir)
    structure = Structure.from_file(cif_path)
else:
    # Demo structure: CsCl Pm-3m
    structure = Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(4.5), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )

# Persist the current input structure for reuse in viewer controls
st.session_state["input_structure"] = structure

with st.sidebar:
    st.subheader("3D Viewer")
    viewer_options = ["Input structure"]
    if "relaxed_structure" in st.session_state:
        viewer_options.append("Relaxed structure")
    default_choice = st.session_state.get("viewer_selection", viewer_options[0])
    if default_choice not in viewer_options:
        default_choice = viewer_options[0]
    selected = st.radio(
        "Structure to display",
        options=viewer_options,
        index=viewer_options.index(default_choice),
    )
    st.session_state["viewer_selection"] = selected

selected_label = st.session_state.get("viewer_selection", "Input structure")
if selected_label == "Relaxed structure" and "relaxed_structure" in st.session_state:
    viewer_structure = st.session_state["relaxed_structure"]
    caption_label = "Relaxed structure"
else:
    viewer_structure = structure
    caption_label = "Input structure"

st.caption(f"Current structure ({caption_label}):")
st.code(lattice_caption(viewer_structure))
render_structure_viewer(viewer_structure, height=480)

tab_relax, tab_md, tab_sp = st.tabs(["🔧 Relaxation", "🏃 Molecular Dynamics", "⚡ Single-Point Energy"])

# ---------------------------
# Relaxation tab (M3GNet Relaxer)
# ---------------------------
with tab_relax:
    st.subheader("Structure Relaxation (M3GNet Relaxer)")
    c1, c2 = st.columns(2)
    with c1:
        fmax = st.number_input("Force threshold fmax (eV/Å)", value=0.01, min_value=0.001, step=0.001, format="%.3f")
    with c2:
        run_relax = st.button("▶️ Relax")

    if run_relax:
        relaxer = Relaxer(potential=pot)
        with st.spinner("Running M3GNet relaxation (CPU)…"):
            results = relaxer.relax(structure, fmax=float(fmax))
        final_structure: Structure = results["final_structure"]
        final_energy = float(results["trajectory"].energies[-1])

        st.success("Relaxation complete.")
        st.markdown("**Final structure:**")
        st.code(lattice_caption(final_structure))
        st.metric("Total energy (eV)", f"{final_energy:.6f}")
        st.metric("Energy per atom (eV/atom)", f"{final_energy/len(final_structure):.6f}")

        out_cif = os.path.join(workdir, "relaxed_structure.cif")
        final_structure.to(fmt="cif", filename=out_cif)
        with open(out_cif, "rb") as f:
            st.download_button("⬇️ Download relaxed CIF", f, file_name="relaxed_structure.cif")

        st.session_state.relaxed_structure = final_structure

# ---------------------------
# MD tab (ASE Langevin + PESCalculator)
# ---------------------------
with tab_md:
    st.subheader("Molecular Dynamics (ASE Langevin + M3GNet)")
    use_relaxed = st.checkbox("Start from relaxed (if available)", value=True)
    start_struct = st.session_state.get("relaxed_structure", structure) if use_relaxed else structure

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        T = st.number_input("Temperature (K)", value=300, min_value=1)
    with c2:
        steps = st.number_input("Steps", value=500, min_value=10, step=50)
    with c3:
        dt_fs = st.number_input("Timestep (fs)", value=1.0, min_value=0.1, step=0.1, format="%.1f")
    with c4:
        friction = st.number_input("Friction (1/ps)", value=0.01, min_value=0.0, step=0.01, format="%.2f")
    run_md = st.button("▶️ Run MD")

    if run_md:
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(start_struct)
        atoms.set_calculator(PESCalculator(pot))

        MaxwellBoltzmannDistribution(atoms, temperature_K=float(T), force_temp=True)

        gamma = float(friction) / 1e-12  # 1/ps -> 1/s
        dyn = Langevin(atoms, timestep=float(dt_fs) * units.fs, temperature_K=float(T), friction=gamma)

        positions: List[np.ndarray] = [atoms.get_positions().copy()]
        prog = st.progress(0)
        status = st.empty()
        with st.spinner("Running MD (CPU)…"):
            total = int(steps)
            for i in range(total):
                dyn.run(1)
                positions.append(atoms.get_positions().copy())
                if i % max(1, total // 100) == 0:
                    prog.progress(int((i + 1) / total * 100))
                    status.text(f"Step {i + 1}/{total}")

        prog.progress(100)
        status.text("MD finished.")

        positions_arr = np.stack(positions, axis=0)
        msd, t_ps, png = msd_and_plot(positions_arr, dt_fs=float(dt_fs))
        st.image(png, caption="MSD vs time", use_column_width=True)
        st.metric("Final potential energy (eV)", f"{atoms.get_potential_energy():.6f}")

        msd_csv = "time_ps,msd_A2\n" + "\n".join(f"{t:.6f},{v:.6f}" for t, v in zip(t_ps, msd))
        st.download_button("⬇️ Download MSD (CSV)", data=msd_csv, file_name="msd.csv", mime="text/csv")

# ---------------------------
# Single-Point tab
# ---------------------------
with tab_sp:
    st.subheader("Single-Point Energy (PESCalculator)")
    target_struct = st.session_state.get("relaxed_structure", structure)
    st.caption("Target structure:")
    st.code(lattice_caption(target_struct))

    do_sp = st.button("⚡ Compute energy")
    if do_sp:
        atoms = AseAtomsAdaptor.get_atoms(target_struct)
        atoms.set_calculator(PESCalculator(pot))
        e = atoms.get_potential_energy()  # eV
        st.success("Single-point complete.")
        st.metric("Total energy (eV)", f"{e:.6f}")
        st.metric("Energy per atom (eV/atom)", f"{e/len(target_struct):.6f}")