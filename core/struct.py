from __future__ import annotations
import io
import warnings
from typing import Optional, Tuple

import numpy as np
from pymatgen.core import Structure, Molecule
from pymatgen.core.lattice import Lattice
from pymatgen.io.ase import AseAtomsAdaptor

warnings.simplefilter("ignore")

def lattice_caption(s: Structure) -> str:
    a, b, c = s.lattice.abc
    alpha, beta, gamma = s.lattice.angles
    return (
        f"a,b,c = {a:.3f}, {b:.3f}, {c:.3f} Å | "
        f"α,β,γ = {alpha:.2f}, {beta:.2f}, {gamma:.2f}° | "
        f"natoms={len(s)}"
    )

def parse_uploaded_structure(uploaded) -> Tuple[Optional[Structure | Molecule], Optional[str]]:
    """
    Parse user upload or supply a default structure.
    Returns (pmg object, message)
    """
    if uploaded is not None:
        try:
            text = uploaded.read().decode("utf-8", errors="ignore")
            name = uploaded.name.lower()

            if name.endswith(".cif"):
                return Structure.from_str(text, fmt="cif"), None
            if name.endswith(("poscar", "contcar")) or name in ("poscar", "contcar"):
                return Structure.from_str(text, fmt="poscar"), None
            if name.endswith(".xyz"):
                return Molecule.from_str(text, fmt="xyz"), None

            # Try Structure then Molecule
            for fmt in ("cif", "poscar"):
                try:
                    return Structure.from_str(text, fmt=fmt), None
                except Exception:
                    pass
            try:
                return Molecule.from_str(text, fmt="xyz"), None
            except Exception as exc:
                return None, f"Could not parse uploaded file: {exc}"

        except Exception as exc:
            return None, f"Upload parse error: {exc}"

    # Default diamond Si conventional cell
    lat = Lattice.cubic(5.431)
    s = Structure(lat, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    return s, "No file uploaded. Using default diamond Si conventional cell."

def ensure_periodic_structure(obj):
    if isinstance(obj, Structure):
        return obj
    if isinstance(obj, Molecule):
        box = Lattice.cubic(30.0)
        return Structure.from_sites(obj.sites, lattice=box)
    atoms = AseAtomsAdaptor.get_atoms(obj)
    return AseAtomsAdaptor.get_structure(atoms)

def atoms_from(obj):
    if obj is None:
        return None
    if isinstance(obj, Molecule):
        obj = ensure_periodic_structure(obj)
    if isinstance(obj, Structure):
        return AseAtomsAdaptor.get_atoms(obj)
    return AseAtomsAdaptor.get_atoms(obj)
