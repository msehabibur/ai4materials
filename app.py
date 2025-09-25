from __future__ import annotations
import os, streamlit as st

from core.struct import parse_uploaded_structure, lattice_caption
from tabs.about_tab import about_tab
from tabs.viewer_tab import viewer_tab
from tabs.relax_tab import relax_tab
from tabs.elastic_tab import elastic_tab
from tabs.phonon_tab import phonon_tab
from tabs.md_tab import md_tab

st.set_page_config(page_title="Materials Studio", layout="wide")
st.title("🧪 Materials Studio")

# session state
st.session_state.setdefault("stop_requested", False)
st.session_state.setdefault("tmp_paths", [])
st.session_state.setdefault("relaxed_cif", None)
st.session_state.setdefault("relax_traj_xyz", None)

# Sidebar
with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader(
        "Upload crystal (POSCAR/CONTCAR/CIF/XYZ)",
        type=["cif", "POSCAR", "CONTCAR", "xyz", "poscar", "contcar"],
        accept_multiple_files=False,
        key="sb_upload",
    )

    st.subheader("Potential family")
    model_family = st.radio(
        "Select family", ["CHGNet", "MACE"], index=0, key="sb_family", horizontal=True
    )

    # Family-specific variants
    if model_family == "CHGNet":
        variant = st.selectbox(
            "CHGNet model",
            ["CHGNet v0.4 (default)", "CHGNet (metals)", "CHGNet (oxides)"],
            index=0, key="sb_variant"
        )
    else:
        variant = st.selectbox(
            "MACE model",
            [
                "Auto (mace-models default)",
                "MACE-MP (small)",
                "MACE-MP (medium)",
                "MACE-OFF23 (medium)",
            ],
            index=0, key="sb_variant"
        )

    st.divider()
    st.subheader("Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🛑 Stop", use_container_width=True, key="sb_stop"):
            st.session_state.stop_requested = True
            st.toast("Stop requested.", icon="🛑")
    with c2:
        if st.button("🧹 Clear cache", use_container_width=True, key="sb_clear"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            st.session_state.stop_requested = False
            st.session_state.relaxed_cif = None
            st.session_state.relax_traj_xyz = None
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
    relax_tab(pmg_obj, model_family, variant)

with tab_elastic:
    elastic_tab(pmg_obj)

with tab_phonon:
    phonon_tab(pmg_obj)

with tab_md:
    md_tab(pmg_obj, model_family, variant)
