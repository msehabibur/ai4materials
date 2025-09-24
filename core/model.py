from __future__ import annotations
import warnings
from functools import lru_cache

warnings.simplefilter("ignore")

# MatGL (robust import)
try:
    import matgl
    from matgl.apps.pes import Potential
    _MATGL_ERR = None
except Exception as exc:
    matgl = None
    Potential = None
    _MATGL_ERR = exc

def list_models() -> list[str]:
    if matgl is None:
        return []
    try:
        return list(matgl.get_available_pretrained_models())
    except Exception:
        return []

@lru_cache(maxsize=4)
def _load_potential_cached(name_or_path: str):
    """Internal: cached loader returning Potential or raising."""
    if matgl is None:
        raise RuntimeError(f"MatGL unavailable: {_MATGL_ERR}")
    obj = matgl.load_model(name_or_path)  # may return Potential OR a bare model
    if Potential is None:
        raise RuntimeError("MatGL Potential API unavailable.")
    if isinstance(obj, Potential):
        pot = obj
    else:
        pot = Potential(model=obj, calc_forces=True, calc_stresses=True)
    # try to push to CPU / no-grad mode if applicable (defensive; harmless if no-op)
    try:
        import torch
        torch.set_grad_enabled(False)
        try:
            pot.model.to("cpu")  # some models expose .model
        except Exception:
            pass
    except Exception:
        pass
    return pot

def load_potential(name_or_path: str):
    """
    Returns (Potential, error_message)
    Uses LRU cache to avoid reloading & duplicate memory.
    """
    try:
        return _load_potential_cached(name_or_path), None
    except Exception as exc:
        return None, str(exc)
