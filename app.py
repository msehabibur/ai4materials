from __future__ import annotations
import os as _os

# Keep single-threaded defaults & GraphBolt off (safe on Streamlit Cloud)
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
_os.environ.setdefault("TORCH_NUM_THREADS", "1")
_os.environ.setdefault("DGL_LOAD_GRAPHBOLT", "0")

import streamlit as st
from core.model import list_models, load_potential
from core.struct import parse_uploaded_structure, lattice_caption
from tabs.viewer_tab import viewer_tab
from tabs.relax_tab import relax_tab
from tabs.elastic_tab import elastic_tab
from tabs.phonon_tab import phonon_tab
from tabs.md_tab import md_tab

st.set_page_config(
    page_title="M3GNet Suite",
    layout="wide",
)

st.title("🔬 M3GNet Suite")

# ───────────────────────────────── Sidebar ─────────────────────────────────
with st.sidebar:
    st.header("Inputs")

    # Model dropdown only (no free-text path)
    available = list_models()
    default_model = available[0] if available else "M3GNet-MP-2018.6.1"
    model_name = st.selectbox("Pretrained PES model", [default_model] + [m for m in available if m != default_model])

    uploaded = st.file_uploader(
        "Upload crystal (POSCAR/CONTCAR/CIF/XYZ)",
        type=["cif", "POSCAR", "CONTCAR", "xyz", "poscar", "contcar"],
        accept_multiple_files=False,
    )

# Load model (Potential)
potential, pot_err = load_potential(model_name)
if pot_err:
    st.error(pot_err)

# Parse structure (default if nothing uploaded)
pmg_obj, parse_msg = parse_uploaded_structure(uploaded)
if parse_msg:
    st.info(parse_msg)

# Top summary
if pmg_obj is not None:
    st.caption(lattice_caption(pmg_obj))

# ───────────────────────────────── Tabs ─────────────────────────────────
tab_view, tab_relax, tab_elastic, tab_phonon, tab_md = st.tabs(
    ["👁️ Viewer", "🔧 Energy Optimization", "🧱 Elastic & Mechanical", "🎼 Phonons", "🌡️ MD (NVT)"]
)

with tab_view:
    viewer_tab(pmg_obj)

with tab_relax:
    relax_tab(pmg_obj, potential)

with tab_elastic:
    elastic_tab(pmg_obj)

with tab_phonon:
    phonon_tab(pmg_obj)

with tab_md:
    md_tab(pmg_obj, potential)
