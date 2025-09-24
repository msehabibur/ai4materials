from __future__ import annotations
import streamlit as st
from pymatgen.core import Structure
from core.viewer import render_structure_viewer

_MAX_VIEW_ATOMS = 400  # hard cap for safe viewing

def _maybe_primitive(pmg_obj, make_primitive: bool):
    if make_primitive and isinstance(pmg_obj, Structure):
        try:
            return pmg_obj.get_primitive_structure()
        except Exception:
            return pmg_obj
    return pmg_obj

def viewer_tab(pmg_obj):
    st.subheader("Structure Viewer")

    col1, col2 = st.columns([1,1])
    with col1:
        as_primitive = st.checkbox("Show primitive cell (recommended)", value=True)
    with col2:
        st.caption("Large cells can crash browsers. Primitive keeps it lean.")

    if pmg_obj is None:
        st.info("Upload a crystal in the sidebar to visualize.")
        return

    obj = _maybe_primitive(pmg_obj, as_primitive)

    # enforce a cap
    n = len(obj) if isinstance(obj, Structure) else len(getattr(obj, "sites", []))
    if n > _MAX_VIEW_ATOMS:
        st.warning(f"Cell has {n} atoms. Showing only primitive cell recommended; viewer limited to ≤{_MAX_VIEW_ATOMS}.")
        if isinstance(obj, Structure):
            try:
                obj = obj.get_primitive_structure()
                n2 = len(obj)
                if n2 > _MAX_VIEW_ATOMS:
                    st.error(f"Primitive still has {n2} atoms (> {_MAX_VIEW_ATOMS}). Please upload a smaller cell.")
                    return
            except Exception:
                st.error("Could not reduce to primitive. Please upload a smaller cell.")
                return

    render_structure_viewer(obj, height=520)
