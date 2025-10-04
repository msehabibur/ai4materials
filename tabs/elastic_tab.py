# tabs/elastic_tab.py
from __future__ import annotations

import os
import json
import csv
import glob
import numbers

import numpy as np
import streamlit as st
from monty.json import jsanitize
from pymatgen.core import Structure

from jobflow import run_locally, SETTINGS
from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import MACERelaxMaker

# ---- Try to import FileStore / MemoryStore across jobflow versions ----
FileStore = None
MemoryStore = None
try:
    from jobflow import FileStore as _FS
    FileStore = _FS
except Exception:
    try:
        from jobflow.core.store import FileStore as _FS
        FileStore = _FS
    except Exception:
        try:
            from jobflow.managers.base import FileStore as _FS
            FileStore = _FS
        except Exception:
            FileStore = None

try:
    from jobflow import MemoryStore as _MS
    MemoryStore = _MS
except Exception:
    try:
        from jobflow.core.store import MemoryStore as _MS
        MemoryStore = _MS
    except Exception:
        try:
            from jobflow.managers.base import MemoryStore as _MS
            MemoryStore = _MS
        except Exception:
            MemoryStore = None


class NoOpStore(object):
    """Final fallback to avoid NoneType.connect crashes on odd jobflow builds."""
    def __init__(self):
        self.index = "jobflow"
    def connect(self):  # run_locally may call this
        pass
    def close(self):
        pass
    def write_document(self, *a, **k):
        return None
    def write_many(self, *a, **k):
        return None
    def update(self, *a, **k):
        return None
    def remove_docs(self, *a, **k):
        return None
    def query_one(self, *a, **k):
        return None
    def query(self, *a, **k):
        return []


def _ensure_store():
    """
    Returns (store, store_dir_or_None) and guarantees SETTINGS.JOB_STORE is set
    and has a non-empty 'index' attribute.
    """
    # Prefer FileStore on disk
    if FileStore is not None:
        try:
            store_dir = os.path.abspath("./jobstore_local")
            os.makedirs(store_dir, exist_ok=True)
            try:
                store = FileStore(store_dir, index="jobflow")
            except TypeError:
                store = FileStore(store_dir)
                if not getattr(store, "index", None):
                    setattr(store, "index", "jobflow")
            SETTINGS.JOB_STORE = store
            return store, store_dir
        except Exception:
            pass

    # Fall back to MemoryStore
    if MemoryStore is not None:
        try:
            try:
                store = MemoryStore(index="jobflow")
            except TypeError:
                store = MemoryStore()
                if not getattr(store, "index", None):
                    setattr(store, "index", "jobflow")
            SETTINGS.JOB_STORE = store
            return store, None
        except Exception:
            pass

    # Last resort
    store = NoOpStore()
    SETTINGS.JOB_STORE = store
    return store, None


# ---------- helpers (simple, syntax-safe) ----------
def _to_plain(obj):
    try:
        return jsanitize(obj, strict=False)
    except Exception:
        if hasattr(obj, "as_dict"):
            try:
                return jsanitize(obj.as_dict(), strict=False)
            except Exception:
                return obj
        return obj


def _to_gpa(val):
    if val is None:
        return None
    if isinstance(val, dict):
        out = {}
        for k, v in val.items():
            out[k] = _to_gpa(v)
        return out
    arr = np.array(val, dtype=float)
    # Heuristic: if in Pa, convert to GPa
    try:
        if np.nanmax(np.abs(arr)) > 1.0e5:
            arr = arr * 1.0e-9
    except Exception:
        pass
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def _looks_like_6x6(mat):
    if isinstance(mat, (list, tuple)) and len(mat) == 6:
        try:
            for row in mat:
                if not isinstance(row, (list, tuple)) or len(row) != 6:
                    return False
                for x in row:
                    if not isinstance(x, numbers.Number):
                        return False
            return True
        except Exception:
            return False
    return False


def _maybe_tensor_anywhere(d):
    # Common direct paths
    if isinstance(d, dict):
        et = d.get("elastic_tensor", None)
        if isinstance(et, dict):
            for k in ("ieee_format", "voigt", "matrix", "C_ij", "C"):
                if k in et and _looks_like_6x6(et[k]):
                    return _to_gpa(et[k])
        for k in ("C_ij", "C", "tensor", "elastic", "elastic_voigt"):
            if k in d and _looks_like_6x6(d[k]):
                return _to_gpa(d[k])

    # Deep walk
    def walk(x):
        if _looks_like_6x6(x):
            return _to_gpa(x)
        if isinstance(x, dict):
            for sub in ("output", "data", "result", "results", "elastic_data", "elastic", "metadata"):
                if sub in x:
                    got = walk(x[sub])
                    if got is not None:
                        return got
            for v in x.values():
                got = walk(v)
                if got is not None:
                    return got
        elif isinstance(x, (list, tuple)):
            for v in x:
                got = walk(v)
                if got is not None:
                    return got
        return None

    return walk(d)


def _props_from_any(d):
    if isinstance(d, dict):
        for k in ("derived_properties", "properties"):
            if k in d and isinstance(d[k], dict):
                return _to_gpa(d[k])
        for sub in ("output", "data", "result", "results", "elastic_data", "elastic", "metadata"):
            if sub in d and isinstance(d[sub], dict):
                got = _props_from_any(d[sub])
                if got:
                    return got
    return {}


def _extract_via_store(store, responses, flow_obj):
    # Try resolving flow.output
    try:
        if hasattr(flow_obj.output, "resolve"):
            resolved = flow_obj.output.resolve(store)
            plain = _to_plain(resolved)
            C = _maybe_tensor_anywhere(plain)
            if C is not None:
                P = _props_from_any(plain)
                return np.array(C, float), P, {"scope": "flow.output.resolve(store)"}
    except Exception:
        pass

    # Then per-job outputs / metadata
    items = list(responses.items()) if isinstance(responses, dict) else list(enumerate(responses))
    for _, rec in items:
        out_obj = getattr(rec, "output", None)
        plain = None
        try:
            if hasattr(out_obj, "resolve"):
                plain = _to_plain(out_obj.resolve(store))
            else:
                plain = _to_plain(out_obj)
        except Exception:
            plain = _to_plain(out_obj)
        C = _maybe_tensor_anywhere(plain)
        if C is not None:
            P = _props_from_any(plain)
            return np.array(C, float), P, {"scope": "job.output(resolved)"}

        meta_plain = _to_plain(getattr(rec, "metadata", None))
        C = _maybe_tensor_anywhere(meta_plain)
        if C is not None:
            P = _props_from_any(meta_plain)
            return np.array(C, float), P, {"scope": "job.metadata"}

    return None


def _extract_via_files(store_dir):
    if not store_dir or not os.path.isdir(store_dir):
        return None
    paths = []
    for pat in ("**/*.json", "*.json"):
        paths.extend(glob.glob(os.path.join(store_dir, pat), recursive=True))
    # newest first
    try:
        paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    except Exception:
        pass
    for p in paths[:200]:
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        plain = _to_plain(data)
        C = _maybe_tensor_anywhere(plain)
        if C is not None:
            P = _props_from_any(plain)
            return np.array(C, float), P, {"scope": "filesystem", "path": p}
    return None


def _pack_csv(C, props):
    buf = []
    buf.append(["Elastic tensor C (GPa) — 6×6 (Voigt/IEEE)"])
    for row in C:
        buf.append([("{:.6f}".format(v)) for v in row])
    buf.append([])
    buf.append(["Property", "Value"])
    keys = [
        "k_voigt", "k_reuss", "k_vrh",
        "g_voigt", "g_reuss", "g_vrh",
        "y_mod", "homogeneous_poisson", "universal_anisotropy",
    ]
    for k in keys:
        v = props.get(k, None)
        if isinstance(v, (int, float)):
            buf.append([k, "{:.6f}".format(v)])
        else:
            buf.append([k, "—"])
    # to CSV bytes
    import io as _io
    import csv as _csv
    sio = _io.StringIO()
    w = _csv.writer(sio)
    for row in buf:
        w.writerow(row)
    return sio.getvalue().encode("utf-8")


# ---------- Streamlit tab ----------
def elastic_tab(pmg_obj: Structure | None):
    st.subheader("🧱 Elastic — default flow (MACE)")
    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer first.")
        return

    if "elastic_running" not in st.session_state:
        st.session_state["elastic_running"] = False
    if "elastic_payload" not in st.session_state:
        st.session_state["elastic_payload"] = None

    disabled = st.session_state["elastic_running"]

    col1, col2 = st.columns(2)
    with col1:
        fmax = st.number_input(
            label="Relax fmax (eV/Å)",
            min_value=1e-8,
            max_value=1e-1,
            value=1e-5,
            step=1e-8,
            format="%.0e",
            disabled=disabled,
        )
    with col2:
        allow_cell = st.selectbox(
            "Relax cell?",
            ["Yes (recommended)", "No (fixed cell)"],
            index=0,
            disabled=disabled,
        ).startswith("Yes")

    run_btn = st.button("Run Elastic Workflow", type="primary", disabled=disabled)

    # Show last results if present
    payload = st.session_state["elastic_payload"]
    if (not st.session_state["elastic_running"]) and payload:
        C = np.array(payload["C_GPa"], dtype=float)
        P = payload["props"]
        st.markdown("**Elastic tensor C (GPa, 6×6 Voigt/IEEE)**")
        st.dataframe([[("{:.6f}".format(v)) for v in row] for row in C], use_container_width=True)

        cA, cB, cC = st.columns(3)
        k_vrh = P.get("k_vrh", None)
        g_vrh = P.get("g_vrh", None)
        ymod = P.get("y_mod", None)
        cA.metric("K_VRH (GPa)", "{:.3f}".format(k_vrh) if isinstance(k_vrh, (int, float)) else "—")
        cB.metric("G_VRH (GPa)", "{:.3f}".format(g_vrh) if isinstance(g_vrh, (int, float)) else "—")
        cC.metric("E (GPa)", "{:.3f}".format(ymod) if isinstance(ymod, (int, float)) else "—")

        cD, cE = st.columns(2)
        po = P.get("homogeneous_poisson", None)
        au = P.get("universal_anisotropy", None)
        cD.metric("Poisson ν", "{:.3f}".format(po) if isinstance(po, (int, float)) else "—")
        cE.metric("Anisotropy AU", "{:.3f}".format(au) if isinstance(au, (int, float)) else "—")

        st.download_button("⬇️ JSON", json.dumps(payload, indent=2).encode("utf-8"), "elastic_results.json")
        st.download_button("⬇️ CSV", _pack_csv(C, P), "elastic_results.csv")

    if (not run_btn) or st.session_state["elastic_running"]:
        return

    # Run once
    st.session_state["elastic_running"] = True
    try:
        status = st.status("Building flow…", expanded=True)
        status.write("Preparing makers…")

        store, store_dir = _ensure_store()
        try:
            st.caption("Using store: {}, index={}, dir={}".format(type(store).__name__, getattr(store, "index", None), store_dir or "—"))
        except Exception:
            pass

        bulk_relax = MACERelaxMaker(relax_cell=True, relax_kwargs={"fmax": float(fmax)})
        el_relax = MACERelaxMaker(relax_cell=bool(allow_cell), relax_kwargs={"fmax": float(fmax)})
        maker = ElasticMaker(bulk_relax_maker=bulk_relax, elastic_relax_maker=el_relax)
        flow = maker.make(structure=pmg_obj)

        status.update(label="Running locally…", state="running")
        rs = run_locally(flow, store=store, create_folders=True, ensure_success=True)

        # Extract results
        status.update(label="Extracting results…", state="running")

        got = _extract_via_store(store, rs, flow)
        if (not got) and store_dir:
            got = _extract_via_files(store_dir)
        if not got:
            # deep scan responses as last resort
            whole_plain = _to_plain(rs)
            Ccand = _maybe_tensor_anywhere(whole_plain)
            if Ccand is not None:
                Pcand = _props_from_any(whole_plain)
                got = (np.array(Ccand, float), Pcand, {"scope": "responses-deep-scan"})

        if not got:
            raise RuntimeError("Elastic results not found in job outputs.")

        C, P, meta = got

        # Backfill E, nu if missing
        K = P.get("k_vrh", None)
        G = P.get("g_vrh", None)
        if isinstance(K, (int, float)) and isinstance(G, (int, float)):
            denom = 3.0 * K + G
            if denom != 0.0:
                P.setdefault("y_mod", 9.0 * K * G / denom)
                P.setdefault("homogeneous_poisson", (3.0 * K - 2.0 * G) / (2.0 * denom))

        st.session_state["elastic_payload"] = {
            "C_GPa": np.array(C, dtype=float).tolist(),
            "props": P,
            "meta": meta,
        }

        status.update(label="Done ✅", state="complete")

        # Render results immediately
        payload = st.session_state["elastic_payload"]
        C = np.array(payload["C_GPa"], dtype=float)
        P = payload["props"]
        st.markdown("**Elastic tensor C (GPa, 6×6 Voigt/IEEE)**")
        st.dataframe([[("{:.6f}".format(v)) for v in row] for row in C], use_container_width=True)

        cA, cB, cC = st.columns(3)
        k_vrh = P.get("k_vrh", None)
        g_vrh = P.get("g_vrh", None)
        ymod = P.get("y_mod", None)
        cA.metric("K_VRH (GPa)", "{:.3f}".format(k_vrh) if isinstance(k_vrh, (int, float)) else "—")
        cB.metric("G_VRH (GPa)", "{:.3f}".format(g_vrh) if isinstance(g_vrh, (int, float)) else "—")
        cC.metric("E (GPa)", "{:.3f}".format(ymod) if isinstance(ymod, (int, float)) else "—")

        cD, cE = st.columns(2)
        po = P.get("homogeneous_poisson", None)
        au = P.get("universal_anisotropy", None)
        cD.metric("Poisson ν", "{:.3f}".format(po) if isinstance(po, (int, float)) else "—")
        cE.metric("Anisotropy AU", "{:.3f}".format(au) if isinstance(au, (int, float)) else "—")

        st.download_button("⬇️ JSON", json.dumps(payload, indent=2).encode("utf-8"), "elastic_results.json")
        st.download_button("⬇️ CSV", _pack_csv(C, P), "elastic_results.csv")

    except Exception as e:
        st.error("Elastic workflow failed: {}".format(e))
        with st.expander("Exception", expanded=True):
            st.exception(e)
    finally:
        st.session_state["elastic_running"] = False
