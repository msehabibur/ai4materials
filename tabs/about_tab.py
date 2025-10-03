# tabs/about_tab.py
from __future__ import annotations
import streamlit as st

def about_tab():
    st.header("Materials Studio — What this tool does")
    st.write(
        "Interactive workflows for crystal visualization, structure optimization, phonons, "
        "elastic properties, and molecular dynamics using **force-field engines** "
        "(MACE by default in this app)."
    )

    with st.container(border=True):
        st.markdown("""
**Md Habibur Rahman**  
*School of Materials Engineering, Purdue University, West Lafayette, IN 47907, USA*  
*rahma103@purdue.edu*
""")

    st.markdown("---")
    with st.expander("📋 Features", expanded=True):
        st.markdown("""
- **Upload** a crystal (CIF/POSCAR/XYZ) from the sidebar  
- **Viewer** with unit cell and supercell builder  
- **Structure Optimization** (MACE + ASE optimizer): progress, energy vs step, before/after lattice, downloads  
- **Elastic Properties** (atomate2 force-field flow): elastic tensor + derived moduli (GPa), with downloads  
- **Phonons** (atomate2 force-field flow): DOS & Band structure + CSV/PNG exports  
- **MD** (ASE + MACE): choose **NVE/NVT/NPT**, streamed trajectory, % progress, stop  
- **Memory-friendly**: streaming & minimal retention
""")

    st.markdown("---")
    st.subheader("Model comparison (high-level)")
    st.table({
        "Method": ["MACE", "CHGNet", "M3GNet"],
        "Core idea": [
            "Equivariant message passing + ACE basis",
            "Graph NN with charge-aware features",
            "MatErials 3-body GNN (universal interatomic potential)"
        ],
        "Strengths": [
            "Accuracy + robustness; stable MD; broad coverage",
            "Fast, widely used; strong for oxides/intercalation",
            "Generalizable; decent phonons/relaxations"
        ],
        "Notes": [
            "Default engine in this app",
            "Not enabled by default here",
            "Not included in this app"
        ]
    })
