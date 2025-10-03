from __future__ import annotations
import io
import json
import csv
import os
import glob
import numbers
from typing import Any, Dict, Tuple, Optional, List, Union

import numpy as np
import streamlit as st
from pymatgen.core import Structure
from monty.json import jsanitize

from atomate2.forcefields.flows.elastic import ElasticMaker
from atomate2.forcefields.jobs import MACERelaxMaker
from jobflow import run_locally, SETTINGS

# ---- Try to import a concrete Store implementation from jobflow (many versions) ----
FileStore = None
MemoryStore = None
try:
    from jobflow import FileStore as _FS
    FileStore = _FS
except Exception:
    try:
        from jobflow.core.store import FileStore as _FS  # alt path
        FileStore = _FS
    except Exception:
        try:
            from jobflow.managers.base import FileStore as _FS  # very old path
            FileStore = _FS
        except Exception:
            FileStore = None

try:
    from jobflow import MemoryStore as _MS
    MemoryStore = _MS
except Exception:
    try:
        from jobflow.core.store import MemoryStore as _MS  # alt path
        MemoryStore = _MS
    except Exception:
        try:
            from jobflow.managers.base import MemoryStore as _MS  # very old path
            MemoryStore = _MS
        except Exception:
            MemoryStore = None


# ---- Last-resort store to avoid 'NoneType.connect' crashes (no persistence) ----
class NoOpStore:
    """Minimal stub with the methods run_locally() usually touches."""
    def __init__(self): self._ok = False
    def connect(self): self._ok = True
    def close(self): pass
    def write_document(self, *a, **k): return None
    def write_many(self, *a, **k): return None
    def update(self, *a, **k): return None
    def remove_docs(self, *a, **k): return None
    def query_one(self, *a, **k): return None
    def query(self, *a, **k): return []


def _ensure_store() -> tuple[object, Optional[str]]:
    """
    Guarantee SETTINGS.JOB_STORE is a usable instance.
    Returns (store, store_dir_if_filestore_else_None)
    """
    store_dir = None
    try:
        if FileStore is not None:
            store_dir = os.path.abspath("./jobstore_local")
            os.makedirs(store_dir, exist_ok=True)
            store = FileStore(store_dir)
            SETTINGS.JOB_STORE = store
            return store, store_dir
    except Exception:
        store_dir = None

    try:
        if MemoryStore is not None:
            store = MemoryStore()
            SETTINGS.JOB_STORE = store
            return store, None
    except Exception:
        pass

    store = NoOpStore()
    SETTINGS.JOB_STORE = store
    return store, None


# ---------- utils ----------
def _to_plain(obj: Any) -> Any:
    try:
        return jsanitize(obj, strict=False)
    except Exception:
        if hasattr(obj, "as_dict"):
            try:
                return jsanitize(obj.as_dict(), strict=False)
            except Exception:
                return obj
        return obj


def _to_gpa(val: Any):
    if val is None:
        return None
    if isinstance(val, dict):
        return {k: _to_gpa(v) for k, v in val.items()}
    arr = np.array(val, dtype=float)
    if np.nanmax(np.abs(arr)) > 1e5:  # likely Pascals
        arr = arr * 1e-9
    if arr.ndim == 0:
        return float(arr)
    return arr.tolist()


def _pack_csv(C: np.ndarray, props: dict) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Elastic tensor C (GPa) — 6×6 (Voigt/IEEE)"])
    for row in C:
        w.writerow([f"{v:.6f}" for v in row])
    w.writerow([])
    w.writerow(["Property", "Value"])
    for k in [
        "k_voigt", "k_reuss", "k_vrh",
        "g_voigt", "g_reuss", "g_vrh",
        "y_mod", "homogeneous_poisson", "universal_anisotropy",
    ]:
        v = props.get(k, None)
        w.writerow([k, f"{v:.6f}" if isinstance(v, (int, float)) else "—"])
    return buf.getvalue().encode()


# ---------- robust extraction ----------
def _looks_like_6x6(mat: Any) -> bool:
    if isinstance(mat, (list, tuple)) and len(mat) == 6:
        try:
            if all(isinstance(r, (list, tuple)) and len(r) == 6 for r in mat):
                for r in mat:
                    for x in r:
                        if not isinstance(x, numbers.Number):
                            return False
                return True
        except Exception:
            return False
    return False


def _maybe_tensor_anywhere(d: Any) -> Optional[List[List[float]]]:
    if isinstance(d, dict):
        et = d.get("elastic_tensor")
        if isinstance(et, dict):
            for k in ("ieee_format", "voigt", "matrix", "C_ij", "C"):
                if k in et and _looks_like_6x6(et[k]):
                    return _to_gpa(et[k])
        for k in ("C_ij", "C", "tensor", "elastic", "elastic_voigt"):
            if k in d and _looks_like_6x6(d[k]):
                return _to_gpa(d[k])

    def walk(x: Any) -> Optional[List[List[float]]]:
        if _looks_like_6x6(x):
            return _to_gpa(x)
        if isinstance(x, dict):
            for sub in ("output", "data", "result", "results", "elastic_data", "elastic", "metadata"):
                if sub in x:
                    got = walk(x[sub]);  # noqa: E702
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


def _props_from_any(d: Any) -> dict:
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


def _extract_via_store(store, rs, flow_obj):
    """Resolve flow and job outputs via a real store (if it supports resolve)."""
    try:
        if hasattr(flow_obj.output, "resolve"):
            resolved = flow_obj.output.resolve(store)
            plain = _to_plain(resolved)
            C = _maybe_tensor_anywhere(plain)
            if C is not None:
                P = _props_from_any(plain)
                return np.array(C, float), P, {"found_in": {"scope": "flow.output.resolve(store)"}, "jobs_seen": []}, plain, []
    except Exception:
        pass

    samples = []
    jobs_seen = []
    items = list(rs.items()) if isinstance(rs, dict) else list(enumerate(rs))
    for i, (jid, rec) in enumerate(items):
        jobs_seen.append({"id": str(jid), "name": getattr(rec, "name", None)})
        out_obj = getattr(rec, "output", None)

        try:
            if hasattr(out_obj, "resolve"):
                out_resolved = out_obj.resolve(store)
                plain = _to_plain(out_resolved)
            else:
                plain = _to_plain(out_obj)
        except Exception:
            plain = _to_plain(out_obj)

        if i < 5:
            samples.append({"job_id": str(jid), "name": getattr(rec, "name", None), "output_sample": plain})

        C = _maybe_tensor_anywhere(plain)
        if C is not None:
            P = _props_from_any(plain)
            return np.array(C, float), P, {"found_in": {"job_id": str(jid), "field": "output(resolved)"}, "jobs_seen": jobs_seen}, plain, samples

        meta_obj = getattr(rec, "metadata", None)
        meta_plain = _to_plain(meta_obj)
        C = _maybe_tensor_anywhere(meta_plain)
        if C is not None:
            P = _props_from_any(meta_plain)
            return np.array(C, float), P, {"found_in": {"job_id": str(jid), "field": "metadata"}, "jobs_seen": jobs_seen}, meta_plain, samples

    return None


def _extract_via_files(store_dir: str):
    """Last-resort: scan the FileStore directory for JSON docs and pull a 6×6 tensor."""
    if not store_dir or not os.path.isdir(store_dir):
        return None
    json_paths = []
    for pat in ("**/*.json", "*.json"):
        json_paths.extend(glob.glob(os.path.join(store_dir, pat), recursive=True))
    json_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for p in json_paths[:200]:
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        plain = _to_plain(data)
        C = _maybe_tensor_anywhere(plain)
        if C is not None:
            P = _props_from_any(plain)
            meta = {"found_in": {"scope": "filesystem", "path": p}, "jobs_seen": []}
            return np.array(C, float), P, meta, {"file": p}, []
    return None


# ---------- render ----------
def _render_payload(payload: dict):
    C = np.array(payload["C_GPa"], dtype=float)
    P = payload["props"]

    st.markdown("**Elastic tensor C (GPa, 6×6 Voigt/IEEE)**")
    st.dataframe([[f"{v:.6f}" for v in row] for row in C], use_container_width=True)

    cA, cB, cC = st.columns(3)
    cA.metric("K_VRH (GPa)", f"{P.get('k_vrh', float('nan')):.3f}" if P.get("k_vrh") is not None else "—")
    cB.metric("G_VRH (GPa)", f"{P.get('g_vrh', float('nan')):.3f}" if P.get("g_vrh") is not None else "—")
    cC.metric("E (GPa)",     f"{P.get('y_mod', float('nan')):.3f}" if P.get('y_mod') is not None else "—")

    cD, cE = st.columns(2)
    po = P.get("homogeneous_poisson", None)
    au = P.get("universal_anisotropy", None)
    cD.metric("Poisson ν", f"{po:.3f}" if isinstance(po, (int, float)) else "—")
    cE.metric("Anisotropy AU", f"{au:.3f}" if isinstance(au, (int, float)) else "—")

    json_bytes = json.dumps(payload, indent=2).encode()
    csv_bytes = _pack_csv(C, P)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ JSON", json_bytes, "elastic_results.json")
    d2.download_button("⬇️ CSV", csv_bytes, "elastic_results.csv")


# ---------- tab ----------
def elastic_tab(pmg_obj: Structure | None):
    st.subheader("🧱 Elastic — default flow (MACE)")
    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer first.")
        return

    # State
    st.session_state.setdefault("elastic_running", False)
    st.session_state.setdefault("elastic_payload", None)
    st.session_state.setdefault("elastic_debug", None)

    disabled = st.session_state.elastic_running

    c1, c2 = st.columns(2)
    with c1:
        fmax = st.number_input(
            "Relax fmax (eV/Å)",
            min_value=1e-8,
            max_value=1e-1,
            value=1e-5,
            step=1e-8,
            format="%.0e",
            disabled=disabled,
        )
    with c2:
        allow_cell = st.selectbox(
            "Relax cell?", ["Yes (recommended)", "No (fixed cell)"],
            index=0, disabled=disabled
        ).startswith("Yes")

    run_btn = st.button("Run Elastic Workflow", type="primary", disabled=disabled)

    # Show last results if available
    if st.session_state.elastic_payload and not st.session_state.elastic_running:
        _render_payload(st.session_state.elastic_payload)
        with st.expander("Debug info", expanded=False):
            st.json(st.session_state.elastic_debug or {})

    if not run_btn or st.session_state.elastic_running:
        return

    # === run once ===
    st.session_state.elastic_running = True
    try:
        status = st.status("Building flow…", expanded=True)
        status.write("Preparing makers…")

        # Ensure we have a usable store and make it the active default
        store, store_dir = _ensure_store()

        bulk_relax = MACERelaxMaker(relax_cell=True,  relax_kwargs={"fmax": float(fmax)})
        el_relax   = MACERelaxMaker(relax_cell=bool(allow_cell), relax_kwargs={"fmax": float(fmax)})
        maker = ElasticMaker(bulk_relax_maker=bulk_relax, elastic_relax_maker=el_relax)
        flow = maker.make(structure=pmg_obj)

        status.update(label="Running locally…", state="running")

        # Always pass the active store (prevents NoneType.connect crash)
        rs = run_locally(flow, store=store, create_folders=True, ensure_success=True)

        # Quick job table
        rows = []
        items = list(rs.items()) if isinstance(rs, dict) else list(enumerate(rs))
        for j, r in items:
            nm = (getattr(r, "name", None) or str(j))
            rows.append({"job": str(j)[:8], "name": nm[:80], "error": bool(getattr(r, "error", None))})
        st.dataframe(rows, use_container_width=True)

        status.update(label="Extracting results…", state="running")

        # 1) Try resolving via store API
        got = _extract_via_store(store, rs, flow)
        if not got and store_dir:
            # 2) Filesystem fallback: scan JSON docs under the FileStore
            got = _extract_via_files(store_dir)
        if not got:
            # 3) As a last resort, deep-scan responses object (plain dicts)
            whole_plain = _to_plain(rs)
            C = _maybe_tensor_anywhere(whole_plain)
            if C is not None:
                P = _props_from_any(whole_plain)
                got = (np.array(C, float), P, {"found_in": {"scope": "responses-deep-scan"}, "jobs_seen": []}, None, [])

        if not got:
            raise RuntimeError("Elastic results not found in job outputs.")

        C, P, meta, resolved_plain, samples = got

        # Backfill E, ν if missing
        K, G = P.get("k_vrh"), P.get("g_vrh")
        if isinstance(K, (int, float)) and isinstance(G, (int, float)):
            denom = 3 * K + G
            if denom:
                P.setdefault("y_mod", 9 * K * G / denom)
                P.setdefault("homogeneous_poisson", (3 * K - 2 * G) / (2 * denom))

        payload = {"C_GPa": np.array(C, dtype=float).tolist(), "props": P}
        st.session_state.elastic_payload = payload
        st.session_state.elastic_debug = {
            "found_in": meta.get("found_in"),
            "jobs_seen": meta.get("jobs_seen"),
            "store_kind": type(store).__name__,
            "store_dir": store_dir,
        }

        status.update(label="Done ✅", state="complete")
        _render_payload(payload)

    except Exception as e:
        st.error(f"Elastic workflow failed: {e}")
        with st.expander("Exception", expanded=True):
            st.exception(e)
    finally:
        st.session_state.elastic_running = False
