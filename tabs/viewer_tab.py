from __future__ import annotations
import streamlit as st
from pymatgen.core import Structure
from core.viewer import render_structure_viewer

_MAX_VIEW_ATOMS = 400

def _maybe_primitive(pmg_obj, make_primitive: bool):
    if make_primitive and isinstance(pmg_obj, Structure):
        try:
            return pmg_obj.get_primitive_structure()
        except Exception:
            return pmg_obj
    return pmg_obj

def viewer_tab(pmg_obj):
    st.subheader("Structure Viewer")
    as_primitive = st.checkbox("Show primitive cell (recommended)", value=True)
    if pmg_obj is None:
        st.info("Upload a crystal in the sidebar to visualize.")
        return
    obj = _maybe_primitive(pmg_obj, as_primitive)
    n = len(obj) if isinstance(obj, Structure) else len(getattr(obj, "sites", []))
    if n > _MAX_VIEW_ATOMS:
        st.warning(f"Cell has {n} atoms. Viewer limited to ≤{_MAX_VIEW_ATOMS}.")
        try:
            obj = obj.get_primitive_structure()
            if len(obj) > _MAX_VIEW_ATOMS:
                st.error("Please upload a smaller/primitive cell.")
                return
        except Exception:
            st.error("Cannot reduce to primitive; upload a smaller cell.")
            return
    render_structure_viewer(obj, height=520)
