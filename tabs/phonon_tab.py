# tabs/phonon_tab.py
from __future__ import annotations

import io
import csv
import json
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from pymatgen.core import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos
from pymatgen.phonon.plotter import PhononBSPlotter, PhononDosPlotter
from atomate2.forcefields.flows.phonons import PhononMaker
from jobflow import run_locally, SETTINGS
from monty.json import jsanitize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PH_JSON = "phonon_json_bytes"
_PH_DOS_PNG = "phonon_dos_png"
_PH_BS_PNG  = "phonon_bs_png"
_PH_DOS_CSV = "phonon_dos_csv"


# ------------------------ Progress helper ------------------------
class _StepProgress:
    """Lightweight progress & status helper for Streamlit."""
    def __init__(self, total_steps: int, label: str = "Working…"):
        self.total = max(int(total_steps), 1)
        self.curr = 0
        self._pct_box = st.empty()
        self._status  = st.empty()
        self._bar     = st.progress(0, text=label)

    def tick(self, msg: str):
        self.curr = min(self.curr + 1, self.total)
        pct = int(self.curr / self.total * 100)
        self._status.write(f"**{msg}**")
        self._bar.progress(pct)
        self._pct_box.caption(f"{self.curr} / {self.total} steps")

    def finish(self, success: bool = True):
        if success:
            self._bar.progress(100, text="Done")
        # If not success, we leave the bar as-is (error will be shown by caller)


# ------------------------ Utilities ------------------------
def _png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _dos_csv_bytes(ph_dos: PhononDos) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Frequency (THz)", "Total DOS"])
    freqs = ph_dos.frequencies
    total = ph_dos.densities
    for f, d in zip(freqs, total):
        w.writerow([f, d])
    return buf.getvalue().encode()


def _to_plain(obj):
    """Monty-sanitize to plain python (dict/list/str/num)."""
    try:
        return jsanitize(obj, strict=False)
    except Exception:
        return obj


def _deep_find(d: Any) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Search arbitrarily nested dict/list structure for objects with keys
    'phonon_dos' and 'phonon_bandstructure'. Returns two dicts (dos_dict, bs_dict)
    or (None, None) if not found.
    """
    dos_obj, bs_obj = None, None

    def walk(x: Any):
        nonlocal dos_obj, bs_obj
        if isinstance(x, dict):
            if "phonon_dos" in x and isinstance(x["phonon_dos"], dict) and dos_obj is None:
                dos_obj = x["phonon_dos"]
            if "phonon_bandstructure" in x and isinstance(x["phonon_bandstructure"], dict) and bs_obj is None:
                bs_obj = x["phonon_bandstructure"]
            for k in ("output", "result", "results", "data", "metadata"):
                if k in x:
                    walk(x[k])
            for v in x.values():
                if dos_obj is not None and bs_obj is not None:
                    return
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                if dos_obj is not None and bs_obj is not None:
                    return
                walk(v)

    walk(d)
    return dos_obj, bs_obj


# ------------------------ Main tab ------------------------
def phonon_tab(pmg_obj: Structure | None):
    st.subheader("🎵 Phonons — default force-field flow (MACE)")
    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer to compute phonons.")
        return

    c1, c2 = st.columns(2)
    with c1:
        min_length = st.number_input(
            "Supercell min. length (Å)",
            min_value=8.0,
            max_value=40.0,
            value=15.0,
            step=1.0,
            key="ph_min_len",
        )
    with c2:
        store_fc = st.checkbox("Store force constants", value=False, key="ph_store_fc")

    run_btn = st.button("Run Phonon", type="primary", key="ph_run_btn")

    # Show persisted outputs if any
    if st.session_state.get(_PH_DOS_PNG):
        st.image(st.session_state[_PH_DOS_PNG], caption="Phonon DOS (PNG)", use_column_width=True)
    if st.session_state.get(_PH_BS_PNG):
        st.image(st.session_state[_PH_BS_PNG], caption="Phonon Band Structure (PNG)", use_column_width=True)
    if st.session_state.get(_PH_DOS_CSV):
        st.download_button("⬇️ DOS (CSV)",
                           st.session_state[_PH_DOS_CSV],
                           "phonon_dos.csv",
                           key="ph_dl_dos_csv_persist")

    if not run_btn:
        return

    # We count 10 logical steps to show progress across the workflow
    steps = _StepProgress(total_steps=10, label="Phonon workflow running…")

    try:
        # 1) Build the flow
        steps.tick("Generating phonon workflow")
        flow = PhononMaker(
            min_length=float(min_length),
            store_force_constants=bool(store_fc),
        ).make(structure=pmg_obj)

        # 2) Run locally (long step)
        steps.tick("Launching local run (this may take a while)")
        rs = run_locally(flow, create_folders=True, ensure_success=False)

        # 3) Convert result to plain python
        steps.tick("Collecting in-memory results")
        rs_plain = _to_plain(rs)

        # 4) Connect to JobStore
        steps.tick("Connecting to JobStore (if configured)")
        store = SETTINGS.JOB_STORE
        if store is not None:
            try:
                store.connect()
            except Exception:
                # keep going; we'll fall back to the in-memory scan
                pass

        # 5) Query post-processing
        steps.tick("Querying post-processing outputs")
        result: Optional[Dict[str, Any]] = None
        if store is not None:
            candidate_names = [
                "generate_frequencies_eigenvectors",
                "phonon_postprocess",
                "build_phonon_bands",
                "get_frequencies_eigenvectors",
                "generate_phonon_bands",
            ]
            for name in candidate_names:
                try:
                    result = store.query_one(
                        {"name": name},
                        properties=["output.phonon_dos", "output.phonon_bandstructure"],
                        load=True,
                        sort={"completed_at": -1},
                    )
                except Exception:
                    result = None
                if result and isinstance(result, dict):
                    break

        # 6) Extract dicts for DOS/BS (store or fallback)
        dos_dict: Optional[dict] = None
        bs_dict: Optional[dict] = None
        if result and isinstance(result, dict):
            out = result.get("output") or {}
            if isinstance(out, dict):
                if isinstance(out.get("phonon_dos"), dict):
                    dos_dict = out["phonon_dos"]
                if isinstance(out.get("phonon_bandstructure"), dict):
                    bs_dict = out["phonon_bandstructure"]

        if dos_dict is None and bs_dict is None:
            steps.tick("Falling back: deep scan of results tree")
            dos_dict, bs_dict = _deep_find(rs_plain)
        else:
            steps.tick("Found outputs in JobStore")

        if dos_dict is None and bs_dict is None:
            raise RuntimeError(
                "Could not locate phonon outputs. Neither JobStore query nor in-memory scan "
                "found 'phonon_dos' or 'phonon_bandstructure'. "
                "Possible causes: flow failed upstream, different job names in this atomate2 version, "
                "or store backend not persisting outputs."
            )

        # 7) Deserialize
        steps.tick("Deserializing DOS / bandstructure")
        ph_dos = PhononDos.from_dict(dos_dict) if isinstance(dos_dict, dict) else None
        ph_bs  = PhononBandStructureSymmLine.from_dict(bs_dict) if isinstance(bs_dict, dict) else None

        if ph_dos is None and ph_bs is None:
            raise RuntimeError("Found phonon keys, but could not deserialize DOS or band structure objects.")

        # 8) Plot DOS
        steps.tick("Plotting DOS")
        dos_png = None
        dos_csv = None
        if ph_dos is not None:
            dos_plot = PhononDosPlotter()
            dos_plot.add_dos("Phonon DOS", ph_dos)
            ax_dos = dos_plot.get_plot()
            fig_dos = ax_dos.get_figure()
            dos_png = _png_bytes(fig_dos)
            dos_csv = _dos_csv_bytes(ph_dos)
            st.session_state[_PH_DOS_PNG] = dos_png
            st.session_state[_PH_DOS_CSV] = dos_csv

        # 9) Plot Band Structure
        steps.tick("Plotting band structure")
        bs_png = None
        if ph_bs is not None:
            bs_plot = PhononBSPlotter(ph_bs)
            ax_bs = bs_plot.get_plot()
            fig_bs = ax_bs.get_figure()
            bs_png = _png_bytes(fig_bs)
            st.session_state[_PH_BS_PNG] = bs_png

        # 10) Store compact JSON bundle
        steps.tick("Storing compact JSON bundle")
        bundle = {
            "has_dos": ph_dos is not None,
            "has_bandstructure": ph_bs is not None,
        }
        st.session_state[_PH_JSON] = json.dumps(bundle).encode()

        steps.finish(success=True)

        # Render
        st.success("Phonon calculation finished ✅")
        if dos_png:
            st.image(dos_png, caption="Phonon DOS (PNG)", use_column_width=True)
            st.download_button("⬇️ DOS (CSV)", dos_csv, "phonon_dos.csv", key="ph_dl_dos_csv")
        if bs_png:
            st.image(bs_png, caption="Phonon Band Structure (PNG)", use_column_width=True)

        if (not dos_png) and (not bs_png):
            st.warning("Phonon run completed but no plots were produced (no DOS/BS available).")

    except Exception as e:
        steps.finish(success=False)
        st.error(f"Phonon workflow failed: {e}")
        with st.expander("Details"):
            st.exception(e)


__all__ = ["phonon_tab"]
