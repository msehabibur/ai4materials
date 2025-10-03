# tabs/leaderboard_tab.py
from __future__ import annotations
import streamlit as st

def leaderboard_tab():
    st.subheader("Materials Leaderboard & MACE MLFF")

    st.markdown("**Leaderboard (Matbench Discovery)**")
    st.link_button("Open Matbench Discovery Leaderboard", "https://matbench-discovery.materialsproject.org/")

    st.divider()
    st.markdown("### What is MACE (Message Passing Atomic Cluster Expansion)?")
    st.markdown(
        "- **Type:** Machine-Learned Force Field (MLFF) based on equivariant message passing.\n"
        "- **Goal:** DFT-level energies/forces at a tiny fraction of DFT cost.\n"
        "- **Strengths:** Good transferability across chemistries, stable MD, strong relaxations.\n"
        "- **Limits:** Still an approximation; extreme charge transfer/magnetism/oxidation states may need careful validation."
    )

    st.divider()
    st.markdown("### MACE vs. CHGNet vs. M3GNet (quick view)")
    st.table({
        "Method": ["MACE", "CHGNet", "M3GNet"],
        "Core idea": [
            "Equivariant message passing + ACE basis",
            "Graph NN with charge-aware features",
            "MatErials 3-body GNN (universal interatomic potential)"
        ],
        "Strengths": [
            "Accuracy + robustness; stable MD; broad coverage",
            "Fast, widely used; good for oxides/intercalation",
            "Generalizable; decent phonons/relaxations"
        ],
        "Notes": [
            "Default engine in this app",
            "Removed from this app (MACE-only)",
            "Not included in this app"
        ]
    })
