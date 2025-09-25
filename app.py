# app.py (SAFE BOOT VERSION)
from __future__ import annotations
import os
import importlib
import traceback
import streamlit as st

# --- Page setup early so UI renders even if later imports fail ---
st.set_page_config(page_title="Materials Studio", layout="wide")
st.title("🧪 Materials Studio")

# ---------------- Session state defaults ----------------
st.session_state.setdefault("stop_requested", False)
st.session_state.setdefault("tmp_paths", [])
st.session_state.setdefault("relaxed_cif", None)
st.session_state.setdefault("relax_traj_xyz", None)

# ---------------- Helper: safe import wrapper ----------------
def _safe_import(module_name: str):
    """
    Import a module lazily and return (module, error_str_or_None).
    Never raises; on failure returns (None, traceback_str).
    """
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:
        tb = "".join(traceback.format_exception(exc))
        return None, tb

# ---------------- Try to import lightweight struct helpers ----------------
parse_uploaded_structure = None
lattice_caption = None
_core_struct, err_struct = _safe_import("core.struct")
if _core_struct:
    parse_uploaded_structure = getattr(_core_struct, "parse_uploaded_structure", None)
    lattice_caption = getattr(_core_struct, "lattice_caption", None)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Inputs")

    uploaded = st.file_uploader(
        "Upload crystal (POSCAR/CONTCAR/CIF/XYZ)",
        type=["cif", "POSCAR", "CONTCAR", "xyz", "poscar", "contcar"],
        accept_multiple_files=False,
        key="sb_upload",
    )

    st.subheader("Potential family")
    model_family = st.radio(
        "Select family", ["CHGNet", "MACE"], index=0, key="sb_family", horizontal=True
    )

    if model_family == "CHGNet":
        variant = st.selectbox(
            "CHGNet model",
            ["CHGNet v0.4 (default)", "CHGNet (metals)", "CHGNet (oxides)"],
            index=0, key="sb_variant",
        )
    else:
        variant = st.selectbox(
            "MACE model",
            ["Auto (mace-models default)", "MACE-MP (small)", "MACE-MP (medium)", "MACE-OFF23 (medium)"],
            index=0, key="sb_variant",
        )

    st.divider()
    low_mem = st.checkbox("Low-memory mode (recommended on Streamlit Cloud)", value=True, key="sb_lowmem")

    st.subheader("Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🛑 Stop", use_container_width=True, key="sb_stop"):
            st.session_state.stop_requested = True
            st.toast("Stop requested.", icon="🛑")
    with c2:
        if st.button("🧹 Clear cache", use_container_width=True, key="sb_clear"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
            except Exception:
                pass
            st.session_state.stop_requested = False
            st.session_state.relaxed_cif = None
            st.session_state.relax_traj_xyz = None
            # cleanup temp files created by tabs
            for p in st.session_state.tmp_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
            st.session_state.tmp_paths = []
            st.success("Cleared caches & temp files.")

# ---------------- Diagnostics panel ----------------
with st.expander("🔎 Diagnostics", expanded=False):
    st.write("If a tab shows an error, expand it to see the traceback. "
             "Cloud logs (Manage app → Cloud logs) will also show details.")
    if err_struct:
        st.warning("`core.struct` failed to import; viewer/parse may be limited.")
        st.code(err_struct)

# ---------------- Parse uploaded structure (if helper available) ----------------
pmg_obj = None
parse_msg = None
if parse_uploaded_structure:
    pmg_obj, parse_msg = parse_uploaded_structure(uploaded)
else:
    if uploaded:
        st.warning("Structure parser not available (`core.struct` import failed).")

if parse_msg:
    st.info(parse_msg)
if lattice_caption and pmg_obj is not None:
    st.caption(lattice_caption(pmg_obj))

# ---------------- Tabs (imports happen lazily inside each block) ----------------
tab_about, tab_view, tab_relax, tab_elastic, tab_phonon, tab_md = st.tabs(
    ["💡 About", "👁️ Viewer", "🧰 Structure Optimization", "🧱 Elastic Properties", "🎼 Phonons", "🌡️ MD"]
)

with tab_about:
    mod, err = _safe_import("tabs.about_tab")
    if err:
        st.error("Failed to load About tab."); st.code(err)
    else:
        try:
            mod.about_tab()
        except Exception as exc:
            st.error("Error running About tab.")
            st.code("".join(traceback.format_exception(exc)))

with tab_view:
    mod, err = _safe_import("tabs.viewer_tab")
    if err:
        st.error("Failed to load Viewer tab."); st.code(err)
    else:
        try:
            mod.viewer_tab(pmg_obj)
        except Exception as exc:
            st.error("Error running Viewer tab.")
            st.code("".join(traceback.format_exception(exc)))

with tab_relax:
    mod, err = _safe_import("tabs.relax_tab")
    if err:
        st.error("Failed to load Structure Optimization tab."); st.code(err)
    else:
        try:
            mod.relax_tab(pmg_obj, model_family, variant, low_mem)
        except Exception as exc:
            st.error("Error running Structure Optimization tab.")
            st.code("".join(traceback.format_exception(exc)))

with tab_elastic:
    mod, err = _safe_import("tabs.elastic_tab")
    if err:
        st.error("Failed to load Elastic Properties tab."); st.code(err)
    else:
        try:
            mod.elastic_tab(pmg_obj, model_family, low_mem)
        except Exception as exc:
            st.error("Error running Elastic Properties tab.")
            st.code("".join(traceback.format_exception(exc)))

with tab_phonon:
    mod, err = _safe_import("tabs.phonon_tab")
    if err:
        st.error("Failed to load Phonons tab."); st.code(err)
    else:
        try:
            mod.phonon_tab(pmg_obj, model_family, low_mem)
        except Exception as exc:
            st.error("Error running Phonons tab.")
            st.code("".join(traceback.format_exception(exc)))

with tab_md:
    mod, err = _safe_import("tabs.md_tab")
    if err:
        st.error("Failed to load MD tab."); st.code(err)
    else:
        try:
            mod.md_tab(pmg_obj, model_family, variant)
        except Exception as exc:
            st.error("Error running MD tab.")
            st.code("".join(traceback.format_exception(exc)))
