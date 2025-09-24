from __future__ import annotations
import os as _os
import warnings

warnings.simplefilter("ignore")

# MatGL (robust import)
try:
    import matgl
    from matgl.apps.pes import Potential
    from matgl.ext.ase import M3GNetCalculator
    _MATGL_ERR = None
except Exception as exc:
    matgl = None
    Potential = None
    M3GNetCalculator = None
    _MATGL_ERR = exc

def list_models() -> list[str]:
    if matgl is None:
        return []
    try:
        return list(matgl.get_available_pretrained_models())
    except Exception:
        return []

def load_potential(name_or_path: str):
    """
    Returns (Potential, error_message)
    """
    if matgl is None:
        return None, f"MatGL unavailable: {_MATGL_ERR}"

    try:
        obj = matgl.load_model(name_or_path)  # may return Potential or bare model
        if Potential is None:
            return None, "MatGL Potential API unavailable."
        if isinstance(obj, Potential):
            return obj, None
        pot = Potential(model=obj, calc_forces=True, calc_stresses=True)
        return pot, None
    except Exception as exc:
        return None, f"Failed to load '{name_or_path}': {exc}"
