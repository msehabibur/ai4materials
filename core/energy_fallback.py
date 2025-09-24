from __future__ import annotations
from typing import Optional
from pymatgen.core.structure import Structure

def direct_ml_energy(structure: Structure, model: str) -> Optional[float]:
    """
    Final fallback: compute total energy (eV) directly with a local ML calculator.
    Tries M3GNet via matgl+ASE first (installed in requirements), then CHGNet if available.

    Returns:
        float energy in eV, or None if calculators are unavailable.
    """
    mdl = (model or "").lower()

    # -------- M3GNet (preferred; included in requirements.txt) --------
    if "m3gnet" in mdl or "m3egnet" in mdl:
        try:
            from matgl.ext.ase import M3GNetCalculator
            from pymatgen.io.ase import AseAtomsAdaptor
            # Use the standard PES model shipped with matgl
            calc = M3GNetCalculator(potential="M3GNet-MP-2021.2.8-PES")
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.calc = calc
            e = atoms.get_potential_energy()  # ASE returns eV
            return float(e)
        except Exception:
            pass  # fall through to CHGNet

    # -------- CHGNet (optional; only works if chgnet is installed) --------
    if "chgnet" in mdl:
        try:
            from chgnet.ext.ase import CHGNetCalculator  # type: ignore
            from pymatgen.io.ase import AseAtomsAdaptor
            calc = CHGNetCalculator()  # default pretrained CHGNet model
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.calc = calc
            e = atoms.get_potential_energy()
            return float(e)
        except Exception:
            pass  # no CHGNet available / failed

    # (Optional) You can add a MACE calculator here if you have a pretrained potential.

    return None