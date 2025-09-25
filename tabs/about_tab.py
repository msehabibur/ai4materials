from __future__ import annotations
import streamlit as st

def about_tab():
    st.header("Materials Studio — What this tool does")
    st.write(
        "Interactive workflows for crystal visualization, structure optimization, phonons, "
        "elastic properties, and molecular dynamics using **CHGNet** and **Atomate2**."
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
- **Viewer** with unit cell  
- **Structure Optimization** (CHGNet + BFGS): progress, stop, energy vs step, before/after lattice, downloads  
- **Elastic Properties** (Atomate2 + CHGNet): elastic tensor + derived moduli (GPa), with units  
- **Phonons** (Atomate2 + phonopy): DOS & Band structure + CSV exports  
- **MD** (ASE + CHGNet): choose **NVE/NVT/NPT**, streamed trajectory, % progress, stop  
- **Memory-friendly**: streaming & minimal retention
""")
