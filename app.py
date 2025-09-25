from __future__ import annotations
import os, shutil, time
import streamlit as st

from core.struct import parse_uploaded_structure, lattice_caption
from tabs.about_tab import about_tab
from tabs.viewer_tab import viewer_tab
from tabs.relax_tab import relax_tab
from tabs.elastic_tab import elastic_tab
from tabs.phonon_tab import phonon_tab
from tabs.md_tab import md_tab

st.set_page_config(page_title="Materials Studio", layout="wide")

# Small UI theme touches (keep fonts normal; plots will set their own font size)
st.title("🧪 Materials Studio")

# Session state for stop/clear
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "tmp_paths" not in st.session_state:
    st.session_state.tmp_paths = []
if "relaxed_cif" not in st.session_state:
    st.session_state.relaxed_cif = None
if "relax_traj_xyz" not in st.session_state:
    st.session_state.relax_traj_xyz = None

# Sidebar
with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader(
        "Upload crystal (POSCAR/CONTCAR/CIF/XYZ)",
        type=["cif", "POSCAR", "CONTCAR", "xyz", "poscar", "contcar"],
        accept_multiple_files=False,
    )
    st.caption("Backend: **CHGNet** (fixed)")

    st.divider()
    st.subheader("Controls")
    colA, colB = st.columns(2)
    with colA:
        if st.button("🛑 Stop", use_container_width=True):
            st.session_state.stop_requested = True
            st.toast("Stop requested.", icon="🛑")
    with colB:
        if st.button("🧹 Clear cache", use_container_width=True):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            st.session_state.stop_requested = False
            st.session_state.relaxed_cif = None
            st.session_state.relax_traj_xyz = None
            # clean any temp files we recorded
            for p in st.session_state.tmp_paths:
                try: os.remove(p)
                except Exception: pass
            st.session_state.tmp_paths = []
            st.success("Cleared caches & temp files.")

pmg_obj, parse_msg = parse_uploaded_structure(uploaded)
if parse_msg:
    st.info(parse_msg)
if pmg_obj is not None:
    st.caption(lattice_caption(pmg_obj))

# Tabs
tab_about, tab_view, tab_relax, tab_elastic, tab_phonon, tab_md = st.tabs(
    ["💡 About", "👁️ Viewer", "🧰 Structure Optimization", "🧱 Elastic Properties", "🎼 Phonons", "🌡️ MD"]
)

with tab_about:
    about_tab()

with tab_view:
    viewer_tab(pmg_obj)

with tab_relax:
    relax_tab(pmg_obj)

with tab_elastic:
    elastic_tab(pmg_obj)

with tab_phonon:
    phonon_tab(pmg_obj)

with tab_md:
    md_tab(pmg_obj)
