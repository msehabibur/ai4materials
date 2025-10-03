from typing import Tuple, List, Optional

from jobflow import run_locally, SETTINGS
from atomate2.forcefields.jobs import ForceFieldRelaxMaker, ForceFieldStaticMaker
from atomate2.forcefields.flows.phonons import PhononMaker

# Optional MD support (depends on your atomate2 build)
try:
    from atomate2.forcefields.jobs import ForceFieldMDMaker  # type: ignore
    HAS_MD = True
except Exception:
    HAS_MD = False

from pymatgen.core.structure import Structure
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
from pymatgen.phonon.dos import PhononDos


# -----------------------
# Robust energy extractor
# -----------------------
def _extract_energy(obj):
    """Recursively find a numeric energy in a dict/list under keys containing 'energy'."""
    from numbers import Number
    if isinstance(obj, dict):
        # common explicit keys first
        for k in ("energy", "final_energy", "total_energy", "e_tot", "e_total"):
            if k in obj and isinstance(obj[k], Number):
                return float(obj[k])
        # fuzzy hit
        for k, v in obj.items():
            if "energy" in k.lower() and isinstance(v, Number):
                return float(v)
        # recurse
        for v in obj.values():
            e = _extract_energy(v)
            if e is not None:
                return e
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            e = _extract_energy(v)
            if e is not None:
                return e
    return None


# ---------------
# Relaxation flow
# ---------------
def make_relax_flow(structure: Structure, model: str, relax_cell: bool = True, fmax: float = 1e-3):
    maker = ForceFieldRelaxMaker(
        force_field_name=model,
        relax_cell=relax_cell,
        relax_kwargs={"fmax": float(fmax)},
    )
    return maker.make(structure=structure)


def run_flow_and_fetch_struct(flow) -> None:
    """Run a Jobflow flow locally; results are written to the Job Store."""
    run_locally(flow, create_folders=True)


def fetch_relax_metrics():
    """Fetch most recent relaxed structure and energy (if present) from the store.

    NOTE: Project ONLY 'output' as a whole to avoid Mongo path collisions.
    """
    store = SETTINGS.JOB_STORE
    store.connect()

    # Try canonical relax doc first
    rec = store.query_one(
        {"name": "force_field_relax"},
        properties=["output", "completed_at", "name"],
        sort={"completed_at": -1},
        load=True,
    )

    # Fallback: any recent doc that has a structure in output
    if rec is None or "output" not in rec or "structure" not in rec["output"]:
        rec = store.query_one(
            {"output.structure": {"$exists": True}},
            properties=["output", "completed_at", "name"],
            sort={"completed_at": -1},
            load=True,
        )

    if rec is None or "output" not in rec or "structure" not in rec["output"]:
        raise RuntimeError("No relaxation results found.")

    structure = Structure.from_dict(rec["output"]["structure"])  # type: ignore
    energy = _extract_energy(rec.get("output", {}))

    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    info = {
        "energy_eV": float(energy) if energy is not None else None,
        "a": a,
        "b": b,
        "c": c,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "natoms": len(structure),
        "source_name": rec.get("name"),
    }
    return structure, info


# ----------------------------
# Static (single-point) energy
# ----------------------------
def make_static_flow(structure: Structure, model: str):
    maker = ForceFieldStaticMaker(force_field_name=model)
    return maker.make(structure=structure)


def fetch_latest_energy() -> Optional[float]:
    """Find the most recent numeric energy in the store, regardless of nesting.

    NOTE: Project ONLY 'output' (no overlapping subpaths).
    """
    store = SETTINGS.JOB_STORE
    store.connect()

    queries = [
        {"output.energy": {"$exists": True}},
        {"output.final_energy": {"$exists": True}},
        {"output.total_energy": {"$exists": True}},
        {"output": {"$exists": True}},
    ]

    for q in queries:
        rec = store.query_one(
            q,
            properties=["output", "completed_at", "name"],
            sort={"completed_at": -1},
            load=True,
        )
        if rec is None:
            continue
        e = _extract_energy(rec.get("output", {}))
        if e is not None:
            return float(e)
    return None


# -------
# Phonons
# -------
def make_phonon_flow(structure: Structure, min_length: float = 15.0, store_force_constants: bool = False):
    maker = PhononMaker(
        min_length=float(min_length),
        store_force_constants=bool(store_force_constants),
    )
    return maker.make(structure=structure)


def fetch_phonon_results() -> Tuple[PhononBandStructureSymmLine, PhononDos]:
    """Fetch latest phonon DOS + bandstructure.

    NOTE: Project ONLY 'output' to avoid path collisions.
    """
    store = SETTINGS.JOB_STORE
    store.connect()

    rec = store.query_one(
        {"name": "generate_frequencies_eigenvectors"},
        properties=["output", "completed_at"],
        sort={"completed_at": -1},
        load=True,
    )

    if rec is None:
        rec = store.query_one(
            {"output.phonon_dos": {"$exists": True}, "output.phonon_bandstructure": {"$exists": True}},
            properties=["output", "completed_at"],
            sort={"completed_at": -1},
            load=True,
        )

    if rec is None:
        raise RuntimeError("No phonon results found.")

    ph_bs = PhononBandStructureSymmLine.from_dict(rec["output"]["phonon_bandstructure"])  # type: ignore
    ph_dos = PhononDos.from_dict(rec["output"]["phonon_dos"])  # type: ignore
    return ph_bs, ph_dos


# --
# MD
# --
def maybe_make_md_flow(structure: Structure, model: str, T: float, dt_fs: float, steps: int):
    if not HAS_MD:
        raise NotImplementedError("Your atomate2 build does not expose ForceFieldMDMaker. Update atomate2.")
    maker = ForceFieldMDMaker(
        force_field_name=model,
        temperature=T,
        timestep=dt_fs,
        steps=int(steps),
        ensemble="nvt",
    )
    return maker.make(structure)


def fetch_md_trajectory() -> Optional[List[Structure]]:
    """Fetch latest MD trajectory. Project ONLY 'output' to avoid collisions."""
    if not HAS_MD:
        return None

    store = SETTINGS.JOB_STORE
    store.connect()

    rec = store.query_one(
        {"name": "force_field_md"},
        properties=["output", "completed_at"],
        sort={"completed_at": -1},
        load=True,
    )

    if rec is None:
        rec = store.query_one(
            {"output.trajectory": {"$exists": True}},
            properties=["output", "completed_at"],
            sort={"completed_at": -1},
            load=True,
        )

    if rec is None:
        return None

    traj = rec["output"].get("trajectory")
    if not traj:
        return None

    return [Structure.from_dict(s) for s in traj]