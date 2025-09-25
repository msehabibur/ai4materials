from __future__ import annotations
from functools import lru_cache

# ---- CHGNet
from chgnet.model import CHGNetCalculator

# ---- MACE
# Preferred: mace-models (easy pretrained access). Fallback: raw MACECalculator with a model file path.
try:
    import mace_models  # pip install mace-models
    _HAVE_MACE_MODELS = True
except Exception:
    _HAVE_MACE_MODELS = False

try:
    from mace.calculators import MACECalculator  # pip install mace-torch
    _HAVE_MACE = True
except Exception:
    _HAVE_MACE = False

# CHGNet variants (wire checkpoints later if you have them)
_CHGNET = {
    "CHGNet v0.4 (default)": {},
    "CHGNet (metals)": {},  # TODO: add tuned kwargs/checkpoints if you have them
    "CHGNet (oxides)": {},  # TODO
}

# MACE variants — we support two strategies:
#  (A) mace-models: load by a friendly name when possible
#  (B) raw MACECalculator: read model path from env var MACE_MODEL_PATH (fallback)
# Note: names below are "friendly". mace-models may map them internally.
_MACE_MODELS = {
    "Auto (mace-models default)": {"loader_name": None},           # mace_models.load()
    "MACE-MP (small)": {"loader_name": "mace_mp_small"},
    "MACE-MP (medium)": {"loader_name": "mace_mp_medium"},
    "MACE-OFF23 (medium)": {"loader_name": "mace_off23_medium"},
}


@lru_cache(maxsize=16)
def get_calculator(model_family: str, variant: str):
    """
    Returns an ASE calculator for the selected family/variant.
    - CHGNet: CHGNetCalculator (pretrained)
    - MACE:   mace-models if available, else MACECalculator(model_path=ENV:MACE_MODEL_PATH)
    """
    if model_family == "MACE":
        if not _HAVE_MACE:
            raise RuntimeError("MACE not installed. Add to requirements: mace-torch")
        # Try mace-models first (super convenient)
        if _HAVE_MACE_MODELS:
            try:
                loader_name = _MACE_MODELS.get(variant, {}).get("loader_name")
                if loader_name is None:
                    model = mace_models.load()  # default model
                else:
                    model = mace_models.load(loader_name)
                calc = model.get_calculator()  # ASE calculator
                return calc
            except Exception:
                # fall through to raw MACECalculator
                pass
        # Raw MACECalculator expects a .model file path
        import os
        mp = os.environ.get("MACE_MODEL_PATH", "").strip()
        if not mp:
            raise RuntimeError(
                "Set environment variable MACE_MODEL_PATH to a .model checkpoint "
                "or enable mace-models. See MACE docs."
            )
        return MACECalculator(model_paths=mp, device="cpu")
    else:
        # CHGNet
        cfg = _CHGNET.get(variant, {})
        return CHGNetCalculator(**cfg)
