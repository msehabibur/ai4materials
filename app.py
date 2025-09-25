# app.py
from __future__ import annotations

import io
import os
import streamlit as st
from pymatgen.core import Structure

# --- Tabs (make sure these files exist under ./tabs) ---
from tabs.viewer_tab import viewer_tab
from tabs.relax_tab import relax_tab
from tabs.md_tab import md_tab
from tabs.elastic_tab import elastic_tab
from tabs.phonon_tab import phonon_tab
from tabs.leaderboard_tab import leaderboard_tab


# ----------------------------- Page Setup -----------------------------
st.set_page_config(
    page_title="Materials Studio — MACE-only",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small, consistent typography for the whole app (plots control their own fonts)
st.markdown("""
<style>
    :root { --base-font-size: 14px; }
    html, body, [class*="css"]  { font-size: var(--base-font-size) !important; }
    .stDownloadButton, .stButton>button { border-radius: 10px; padding: 0.4rem 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ----------------------------- Helpers -----------------------------
def _guess_fmt(name: str) -> str:
    n = (name or "").lower()
    if n.endswith(".cif"):
        return "cif"
    if n.endswith("poscar") or n == "poscar":
        return "poscar"
    if n.endswith(".xyz"):
        return "xyz"
    # default try CIF; most common for this app
    return "cif"


def _load_structure_from_sidebar() -> Structure | None:
    with st.sidebar:
        st.header("📁 Structure")
        up = st.file_uploader(
            "Upload CIF / POSCAR / XYZ",
            type=["cif", "poscar", "vasp", "xyz"],
            key="sb_upl",
            help="Upload your crystal file. You can expand it as a supercell inside the Viewer tab.",
        )

        pmg_obj = None
        if up is not None:
            try:
                fmt = _guess_fmt(up.name)
                # Try binary-safe decoding; fall back to raw then decode
                raw = up.getvalue()
                try:
                    text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    text = io.BytesIO(raw).read().decode("utf-8", errors="ignore")
                pmg_obj = Structure.from_str(text, fmt=fmt)
                st.success(f"Loaded structure: {len(pmg_obj)} atoms")
            except Exception as e:
                st.error(f"Failed to parse structure ({up.name}): {e}")
        return pmg_obj


def _about_tab():
    st.subheader("About this Tool")
    st.markdown(
        "This app uses **MACE** (Message Passing Atomic Cluster Expansion) as the default and only "
        "Machine-Learned Force Field to run **structure optimization**, **molecular dynamics**, "
        "and **phonons**, plus **elastic constants** via atomate2 force-field workflows (MACE relaxers)."
    )
    st.markdown(
        "- **Viewer & Supercell:** Inspect and expand your crystal before simulations.\n"
        "- **Structure Optimization:** Choose optimizer (FIRE/BFGS), set force tolerance, choose free/fixed volume.\n"
        "- **Molecular Dynamics:** Select ensemble (NVE/NVT/NPT), temperatures, timestep, and save stride; download trajectories.\n"
        "- **Elastic Constants:** MACE-only elastic workflow; outputs Cᵢⱼ (GPa), K/G/E (GPa), ν, AU, with JSON/CSV downloads.\n"
        "- **Phonons:** Force-field phonons with compact, fixed-size plots."
    )
    st.info(
        "For large batches or heavy systems, consider deploying the backend with queued workers (FastAPI + Redis/RQ + MinIO) "
        "so the UI remains responsive and memory usage stays predictable."
    )


# ----------------------------- Main -----------------------------
def main():
    st.title("🧪 Materials Studio — MACE-only")

    pmg_obj = _load_structure_from_sidebar()

    tabs = st.tabs([
        "👁 Viewer",
        "🧰 Structure Optimization",
        "🏃 MD",
        "🧱 Elastic Constants",
        "🎵 Phonons",
        "🏆 Leaderboard",
        "ℹ️ About",
    ])

    with tabs[0]:
        viewer_tab(pmg_obj)

    with tabs[1]:
        relax_tab(pmg_obj)

    with tabs[2]:
        md_tab(pmg_obj)

    with tabs[3]:
        elastic_tab(pmg_obj)

    with tabs[4]:
        phonon_tab(pmg_obj)

    with tabs[5]:
        leaderboard_tab()

    with tabs[6]:
        _about_tab()


if __name__ == "__main__":
    main()
