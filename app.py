from __future__ import annotations

import io
import streamlit as st
from pymatgen.core import Structure

# 1) MUST be the first Streamlit call
st.set_page_config(
    page_title="Materials Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2) Environment setup (no st.* inside)
from tabs.utils_env import harden_env, sanitize_logs
harden_env()
sanitize_logs()

# 3) Optional lightweight CSS
st.markdown("""
<style>
  :root { --base-font-size: 14px; }
  html, body, [class*="css"]  { font-size: var(--base-font-size) !important; }
  .stDownloadButton, .stButton>button { border-radius: 10px; padding: 0.4rem 0.8rem; }
</style>
""", unsafe_allow_html=True)

# 4) Defer heavy tab imports until after page config
from tabs.viewer_tab import viewer_tab
from tabs.relax_tab import relax_tab
from tabs.md_tab import md_tab
from tabs.elastic_tab import elastic_tab
from tabs.phonon_tab import phonon_tab
from tabs.about_tab import about_tab


# ---------- helpers ----------
def _guess_fmt(name: str) -> str:
    n = (name or "").lower()
    if n.endswith(".cif"):
        return "cif"
    if n.endswith(".xyz"):
        return "xyz"
    if n.endswith("poscar") or n.endswith("contcar") or n == "poscar" or n == "contcar":
        return "poscar"
    return "cif"


def _load_structure_from_sidebar() -> Structure | None:
    with st.sidebar:
        st.header("📁 Structure")
        up = st.file_uploader(
            "Upload CIF / POSCAR / XYZ",
            type=["cif", "poscar", "vasp", "xyz"],
            key="sb_upl",
            help="Upload your crystal file. You can expand it as a supercell in the Viewer tab.",
        )

        pmg_obj = None
        if up is not None:
            try:
                fmt = _guess_fmt(up.name)
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


# ---------- main ----------
def main():
    st.title("🧪 Materials Studio")
    pmg_obj = _load_structure_from_sidebar()

    tabs = st.tabs([
        "👁 Viewer",
        "🧰 Structure Optimization",
        "🏃 MD",
        "🧱 Elastic",
        "🎵 Phonons",
        "ℹ️ About",
    ])

    with tabs[0]: viewer_tab(pmg_obj)
    with tabs[1]: relax_tab(pmg_obj)
    with tabs[2]: md_tab(pmg_obj)
    with tabs[3]: elastic_tab(pmg_obj)
    with tabs[4]: phonon_tab(pmg_obj)
    with tabs[5]: about_tab()


if __name__ == "__main__":
    main()
