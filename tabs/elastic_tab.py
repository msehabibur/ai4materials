# tabs/elastic_tab.py
from __future__ import annotations
import io, json, csv, traceback
import numpy as np
import streamlit as st
from pymatgen.core import Structure
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import MACERelaxMaker
from jobflow import run_locally, SETTINGS

_EL_JSON = "elastic_json_bytes"
_EL_CSV  = "elastic_csv_bytes"
_EL_LAST = "elastic_last_payload"

def _to_gpa(x):
    """Convert Pa→GPa if values look too large; pass through if already GPa."""
    if x is None:
        return None
    # Works for dicts, lists, scalars
    if isinstance(x, dict):
        return {k: _to_gpa(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.array(x, dtype=float)
        # Heuristic: if any abs > 1e5, assume Pa→GPa
        needs = np.any(np.abs(arr) > 1e5)
        factor = 1e-9 if needs else 1.0
        return (arr * factor).tolist()
    # scalar
    v = float(x)
    factor = 1e-9 if abs(v) > 1e5 else 1.0
    return v * factor

def _calc_iso_from_kg(k_gpa: float, g_gpa: float):
    """Return (E, nu) from isotropic K,G in GPa; handle edge cases."""
    K = float(k_gpa); G = float(g_gpa)
    denom = (3 * K + G)
    if abs(denom) < 1e-8:
        return None, None
    E = 9 * K * G / denom
    nu = (3 * K - 2 * G) / (2 * denom)
    return E, nu

def _pack_csv(c_ij_gpa, props):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Elastic tensor C_ij (GPa) 6x6"])
    for row in c_ij_gpa:
        w.writerow([f"{v:.3f}" for v in row])
    w.writerow([])
    w.writerow(["Property", "Value (GPa or –)"])
    keys = [
        ("k_voigt", "Bulk K_V (GPa)"),
        ("k_reuss", "Bulk K_R (GPa)"),
        ("k_vrh",   "Bulk K_VRH (GPa)"),
        ("g_voigt", "Shear G_V (GPa)"),
        ("g_reuss", "Shear G_R (GPa)"),
        ("g_vrh",   "Shear G_VRH (GPa)"),
        ("y_mod",   "Young's modulus (GPa)"),
        ("homogeneous_poisson", "Poisson ratio (–)"),
        ("universal_anisotropy", "Universal anisotropy (–)"),
    ]
    for k, label in keys:
        val = props.get(k, None)
        w.writerow([label, f"{val:.3f}" if isinstance(val, (int, float)) and val is not None else "—"])
    return buf.getvalue().encode()

def elastic_tab(pmg_obj: Structure | None):
    st.subheader("Elastic Constants — MACE")
    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer to compute elastic constants.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fmax = st.number_input(
            "Force tolerance fmax (eV/Å)",
            value=1e-5, min_value=1e-6, max_value=1e-2, step=1e-5, format="%.0e",
            key="el_fmax",
        )
    with c2:
        relax_cell = st.selectbox(
            "Bulk relax allows cell change?",
            ["Relax cell (recommended)", "Keep cell fixed"],
            key="el_relax_cell",
        ).startswith("Relax")

    run_btn = st.button("Run Elastic Workflow", type="primary", key="el_run_btn")

    # show previous results if available
    if st.session_state.get(_EL_LAST):
        last = st.session_state[_EL_LAST]
        st.success("Showing previous elastic results.")
        _render_results(last)

    if not run_btn:
        return

    try:
        progress = st.progress(0, text="Building workflow… 10%")
        bulk_maker = MACERelaxMaker(
            relax_cell=True,
            relax_kwargs={"fmax": float(fmax)}
        )
        el_maker = MACERelaxMaker(
            relax_cell=False if not relax_cell else True,  # if relax_cell False → keep fixed
            relax_kwargs={"fmax": float(fmax)}
        )
        flow = ElasticMaker(
            bulk_relax_maker=bulk_maker,
            elastic_relax_maker=el_maker,
        ).make(structure=pmg_obj)

        progress.progress(20, text="Running locally… 20%")
        _ = run_locally(flow, create_folders=True)

        progress.progress(70, text="Querying results… 70%")
        store = SETTINGS.JOB_STORE
        store.connect()
        result = store.query_one(
            {"name": "fit_elastic_tensor"},
            properties=["output.elastic_tensor", "output.derived_properties"],
            load=True,
            sort={"completed_at": -1},
        )
        if not result or "output" not in result:
            raise RuntimeError("No elastic results found in store.")

        et = result["output"]["elastic_tensor"]
        derived = result["output"]["derived_properties"] or {}

        # Normalize units to GPa
        c_ij = et.get("ieee_format") or et.get("matrix")
        c_ij_gpa = _to_gpa(c_ij)
        props_gpa = _to_gpa(derived)

        # If y_mod missing or seems off, recompute from VRH
        K = props_gpa.get("k_vrh")
        G = props_gpa.get("g_vrh")
        if (props_gpa.get("y_mod") is None) and (K is not None) and (G is not None):
            E, nu = _calc_iso_from_kg(K, G)
            if E is not None:
                props_gpa["y_mod"] = E
            if nu is not None:
                props_gpa["homogeneous_poisson"] = nu

        payload = {"c_ij_gpa": c_ij_gpa, "props_gpa": props_gpa}
        st.session_state[_EL_LAST] = payload

        # Downloads
        st.session_state[_EL_JSON] = json.dumps(payload, indent=2).encode()
        st.session_state[_EL_CSV] = _pack_csv(c_ij_gpa, props_gpa)

        progress.progress(100, text="Done")
        _render_results(payload)

    except Exception as exc:
        st.error(f"Elastic workflow failed: {exc}")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))

def _render_results(payload: dict):
    c_ij = np.array(payload["c_ij_gpa"], dtype=float)
    props = payload["props_gpa"]

    st.markdown("**Elastic tensor C** (GPa, IEEE 6×6):")
    st.dataframe(
        np.array([[f"{v:.3f}" for v in row] for row in c_ij]),
        use_container_width=True,
        hide_index=True,
    )

    # Small, clear metric grid
    k_v, k_r, k_vrh = props.get("k_voigt"), props.get("k_reuss"), props.get("k_vrh")
    g_v, g_r, g_vrh = props.get("g_voigt"), props.get("g_reuss"), props.get("g_vrh")
    y_mod = props.get("y_mod")
    nu = props.get("homogeneous_poisson")
    au = props.get("universal_anisotropy")

    cols = st.columns(3)
    cols[0].metric("Bulk K_V (GPa)", f"{k_v:.3f}" if k_v is not None else "—")
    cols[1].metric("Bulk K_R (GPa)", f"{k_r:.3f}" if k_r is not None else "—")
    cols[2].metric("Bulk K_VRH (GPa)", f"{k_vrh:.3f}" if k_vrh is not None else "—")

    cols = st.columns(3)
    cols[0].metric("Shear G_V (GPa)", f"{g_v:.3f}" if g_v is not None else "—")
    cols[1].metric("Shear G_R (GPa)", f"{g_r:.3f}" if g_r is not None else "—")
    cols[2].metric("Shear G_VRH (GPa)", f"{g_vrh:.3f}" if g_vrh is not None else "—")

    cols = st.columns(3)
    cols[0].metric("Young's E (GPa)", f"{y_mod:.3f}" if y_mod is not None else "—")
    cols[1].metric("Poisson ν (–)", f"{nu:.3f}" if nu is not None else "—")
    cols[2].metric("Anisotropy AU (–)", f"{au:.3f}" if au is not None else "—")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download JSON", st.session_state.get(_EL_JSON, b"{}"), "elastic_results.json", key="el_dl_json")
    with c2:
        st.download_button("Download CSV", st.session_state.get(_EL_CSV, b""), "elastic_results.csv", key="el_dl_csv")

    # If anything is negative in K/G/E, add a hint
    negs = []
    for lbl, v in [("K_VRH", k_vrh), ("G_VRH", g_vrh), ("E", y_mod)]:
        if v is not None and v < 0:
            negs.append(lbl)
    if negs:
        st.warning(
            "Some moduli are **negative** ({}). This usually indicates mechanical instability at the sampled "
            "strain / configuration, insufficient relaxation (increase accuracy / lower fmax), or unit issues. "
            "Units have been normalized to GPa here; if negatives persist, the crystal may be mechanically unstable."
            .format(", ".join(negs)),
            icon="⚠️",
        )
