# app.py
from __future__ import annotations
import streamlit as st
from pymatgen.core import Structure

from tabs.viewer_tab import viewer_tab
from tabs.relax_tab import relax_tab
from tabs.md_tab import md_tab
from tabs.phonon_tab import phonon_tab
from tabs.leaderboard_tab import leaderboard_tab

st.set_page_config(page_title="Materials Studio — MACE", page_icon="🧪", layout="wide")

def _load_structure_from_sidebar() -> Structure | None:
    with st.sidebar:
        st.header("📁 Structure")
        up = st.file_uploader("Upload CIF/POSCAR/XYZ", type=["cif", "poscar", "vasp", "xyz"], key="sb_upl")
        pmg_obj = None
        if up is not None:
            try:
                fmt = "cif"
                name = (up.name or "").lower()
                if name.endswith("poscar") or name == "poscar":
                    fmt = "poscar"
                elif name.endswith(".xyz"):
                    fmt = "xyz"
                content = up.getvalue().decode("utf-8", errors="ignore")
                pmg_obj = Structure.from_str(content, fmt=fmt)
                st.success(f"Loaded structure: {len(pmg_obj)} atoms")
            except Exception as e:
                st.error(f"Failed to parse structure: {e}")
        return pmg_obj

def about_tab():
    st.subheader("About this Tool")
    st.markdown(
        "This app uses **MACE** (Message Passing Atomic Cluster Expansion) as the default and only "
        "Machine-Learned Force Field to run structure optimization, molecular dynamics, and phonons. "
        "All CHGNet/M3GNet options have been removed to ensure a focused, consistent experience."
    )
    st.markdown(
        "- **Viewer & Supercell:** Inspect and expand your crystal before simulations.\n"
        "- **Structure Optimization:** Choose optimizer (FIRE/BFGS), force tolerance, and whether to relax the cell.\n"
        "- **Molecular Dynamics:** Pick ensemble (NVE/NVT/NPT), temperatures, timestep, stride; download trajectories.\n"
        "- **Phonons:** Run atomate2 force-field phonon workflow; plots scaled to a compact fixed space."
    )
    st.info("Tip: Keep systems moderate in size to stay within memory budgets; for large sweeps, use a backend queue/workers setup.")

def main():
    st.title("🧪 Materials Studio — MACE-only")
    pmg_obj = _load_structure_from_sidebar()

    tabs = st.tabs(["👁 Viewer", "🧰 Structure Optimization", "🏃 MD", "🎵 Phonons", "🏆 Leaderboard", "ℹ️ About"])
    with tabs[0]:
        viewer_tab(pmg_obj)
    with tabs[1]:
        relax_tab(pmg_obj)
    with tabs[2]:
        md_tab(pmg_obj)
    with tabs[3]:
        phonon_tab(pmg_obj)
    with tabs[4]:
        leaderboard_tab()
    with tabs[5]:
        about_tab()

if __name__ == "__main__":
    main()
