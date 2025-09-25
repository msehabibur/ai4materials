# tabs/elastic_tab.py
from __future__ import annotations
import gc, traceback, numpy as np, streamlit as st

def _lazy_import():
    try:
        from atomate2.forcefields.jobs import CHGNetRelaxMaker as _CHG
    except Exception:
        _CHG = None
    try:
        from atomate2.forcefields.jobs import MACERelaxMaker as _MACE
    except Exception:
        _MACE = None
    try:
        from atomate2.forcefields.flows.elastic import ElasticMaker
        from jobflow import run_locally, SETTINGS
        return _CHG, _MACE, ElasticMaker, run_locally, SETTINGS, None
    except Exception as exc:
        return None, None, None, None, None, exc

# ----- Unit helpers -----
def _to_gpa_value(x: float) -> float:
    """Normalize a single modulus-like value to GPa based on magnitude."""
    v = float(x)
    a = abs(v)
    if a >= 1e9:   # likely Pa
        return v / 1e9
    if a >= 1e4:   # likely MPa
        return v / 1e3
    return v       # already GPa (or small)

def _tensor_to_gpa(C):
    arr = np.array(C, dtype=float)
    # infer scale from diagonal median
    diag = np.abs(np.diag(arr)) if arr.ndim == 2 else np.abs(arr)
    med = np.median(diag) if diag.size else 1.0
    if med >= 1e9:   s = 1/1e9
    elif med >= 1e4: s = 1/1e3
    else:            s = 1.0
    return arr * s

def _pretty_tensor_GPa(C):
    A = _tensor_to_gpa(C)
    lines = ["Elastic tensor C_ij (GPa):"]
    for row in A:
        lines.append("  " + "  ".join(f"{v:10.3f}" for v in row))
    return "\n".join(lines)

def _normalize_props_to_gpa(props: dict) -> dict:
    """Return copy of props with moduli in GPa, dimensionless untouched."""
    if not props: return {}
    out = {}
    for k, v in props.items():
        if v is None:
            out[k] = None; continue
        if k in ("homogeneous_poisson", "universal_anisotropy"):
            out[k] = float(v)
        else:
            out[k] = _to_gpa_value(float(v))
    return out

def _backstop_recompute_E_nu(props_gpa: dict) -> dict:
    """If E or nu look off/missing, recompute from VRH K,G in GPa."""
    K = props_gpa.get("k_vrh")
    G = props_gpa.get("g_vrh")
    if isinstance(K, (int,float)) and isinstance(G, (int,float)) and (3*K + G) != 0:
        E_calc = 9 * K * G / (3 * K + G)
        nu_calc = (3 * K - 2 * G) / (2 * (3 * K + G))
        # Only overwrite if missing or clearly absurd (e.g., >1e5 GPa)
        E_old = props_gpa.get("y_mod")
        if not isinstance(E_old, (int,float)) or abs(E_old) > 1e5:
            props_gpa["y_mod"] = E_calc
        nu_old = props_gpa.get("homogeneous_poisson")
        if not isinstance(nu_old, (int,float)) or not (-1.0 < nu_old < 0.5):
            props_gpa["homogeneous_poisson"] = nu_calc
    return props_gpa

def _pretty_moduli(props_gpa: dict) -> str:
    fields = [
        ("k_voigt", "Bulk K_V (GPa)"),
        ("k_reuss", "Bulk K_R (GPa)"),
        ("k_vrh",  "Bulk K_VRH (GPa)"),
        ("g_voigt", "Shear G_V (GPa)"),
        ("g_reuss", "Shear G_R (GPa)"),
        ("g_vrh",  "Shear G_VRH (GPa)"),
        ("y_mod",  "Young's modulus (GPa)"),
        ("homogeneous_poisson", "Poisson ratio (–)"),
        ("universal_anisotropy", "Universal anisotropy (–)"),
    ]
    lines, warn_negative = [], False
    for key, label in fields:
        val = props_gpa.get(key, None)
        if val is None: continue
        try:
            v = float(val)
            if "(GPa)" in label and v < 0: warn_negative = True
            lines.append(f"{label}: {v:,.3f}")
        except Exception:
            lines.append(f"{label}: {val}")
    if warn_negative:
        lines += [
            "",
            "⚠️ One or more moduli are negative. This typically indicates mechanical instability,",
            "   insufficient relaxation, or a poor fit. Re-optimize the structure and retry."
        ]
    return "\n".join(lines)

def elastic_tab(pmg_obj, model_family: str, low_mem: bool):
    st.subheader("Elastic Properties")
    CHG, MACE, ElasticMaker, run_locally, SETTINGS, err = _lazy_import()
    if err:
        st.error("Dependencies missing. Ensure: atomate2[phonons,forcefields], jobflow, phonopy, spglib.")
        st.code(str(err)); return

    # allow very fine fmax
    default_fmax = 0.000100 if low_mem else 0.000050
    col1, col2 = st.columns(2)
    with col1:
        fmax = st.number_input("Relax fmax (eV/Å)",
                               value=float(default_fmax), min_value=0.000001, max_value=0.01,
                               step=0.000001, format="%.6f", key="el_fmax")
    with col2:
        run = st.button("Run elastic workflow", type="primary", disabled=(pmg_obj is None), key="el_run")
    if not run: return

    progress = st.progress(0, text="Preparing flow…")
    pct_label = st.empty()

    try:
        fam = (model_family or "CHGNet").strip().lower()
        def _pick_relax_maker(relax_cell: bool):
            if fam == "mace" and MACE is not None:
                return MACE(relax_cell=relax_cell, relax_kwargs={"fmax": float(fmax)})
            if fam == "chgnet" and CHG is not None:
                return CHG(relax_cell=relax_cell, relax_kwargs={"fmax": float(fmax)})
            if MACE is not None:
                return MACE(relax_cell=relax_cell, relax_kwargs={"fmax": float(fmax)})
            if CHG is not None:
                return CHG(relax_cell=relax_cell, relax_kwargs={"fmax": float(fmax)})
            return None  # let ElasticMaker decide

        progress.progress(10, text="Preparing flow…"); pct_label.write("**Progress:** 10%")
        flow = ElasticMaker(
            bulk_relax_maker=_pick_relax_maker(relax_cell=True),
            elastic_relax_maker=_pick_relax_maker(relax_cell=False),
        ).make(structure=pmg_obj)

        progress.progress(30, text="Running deformations…"); pct_label.write("**Progress:** 30%")
        _ = run_locally(flow, create_folders=True)

        progress.progress(85, text="Fitting elastic tensor…"); pct_label.write("**Progress:** 85%")
        store = SETTINGS.JOB_STORE
        store.connect()
        result = store.query_one(
            {"name": "fit_elastic_tensor"},
            properties=["output.elastic_tensor", "output.derived_properties"],
            load=True,
            sort={"completed_at": -1},
        )

        if not result:
            progress.progress(100, text="Done (no results)"); pct_label.write("**Progress:** 100%")
            st.warning("No results found."); return

        et = result["output"]["elastic_tensor"]
        dp = result["output"]["derived_properties"]

        # Tensor pretty print (GPa)
        tensor_to_show = et.get("ieee_format", et)
        st.code(_pretty_tensor_GPa(tensor_to_show))

        # Normalize properties to GPa, then backstop recompute E/nu
        props_gpa = _normalize_props_to_gpa(dp or {})
        props_gpa = _backstop_recompute_E_nu(props_gpa)
        st.code(_pretty_moduli(props_gpa))

        del result, et, dp
        gc.collect()

        progress.progress(100, text="Done"); pct_label.write("**Progress:** 100%")

    except Exception as exc:
        st.error("Elastic workflow failed.")
        with st.expander("Traceback"):
            st.code("".join(traceback.format_exception(exc)))
