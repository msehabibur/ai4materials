import streamlit as st

try:
    import py3Dmol
    HAS_PY3DMOL = True
except Exception:
    HAS_PY3DMOL = False

from pymatgen.core.structure import Structure


def structure_summary(struct: Structure):
    a, b, c = struct.lattice.abc
    alpha, beta, gamma = struct.lattice.angles
    st.caption(
        f"a,b,c = {a:.3f}, {b:.3f}, {c:.3f} Å | "
        f"α,β,γ = {alpha:.2f}, {beta:.2f}, {gamma:.2f}° | "
        f"natoms={len(struct)}"
    )


def structure_viewer_widget(struct: Structure, caption: str = "Structure"):
    if not HAS_PY3DMOL:
        st.info("Install py3Dmol to enable 3D structure viewer (pip install py3Dmol).")
        return
    xyz = struct.to(fmt="xyz")
    view = py3Dmol.view(width=700, height=500)
    view.addModel(xyz, "xyz")
    view.setStyle({"sphere": {"scale": 0.3}, "stick": {"radius": 0.15}})
    view.zoomTo()
    view.show()
    st.components.v1.html(view._make_html(), height=520)
    if caption:
        st.caption(caption)