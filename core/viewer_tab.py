from __future__ import annotations
import streamlit as st
from core.viewer import render_structure_viewer

def viewer_tab(pmg_obj):
    st.subheader("Structure Viewer")
    render_structure_viewer(pmg_obj, height=520)
