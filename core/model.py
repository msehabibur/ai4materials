# core/model.py
from __future__ import annotations
from functools import lru_cache

# CHGNet and MACE are **not** imported at module import time.
# This avoids slow/fragile imports blocking Streamlit from rendering the UI.

# CHGNet variants (extend with checkpoints/kwargs if you have them)
_CHGNET = {
    "CHGNet v0.4 (default)": {},
    "CHGNet (metals)": {},  # TODO: add tuned kwargs/checkpoints if available
    "CHGNet (oxides)": {},  # TODO
}

# MACE variants (friendly names). If mace-models is available, we’ll use it.
_MACE_MODELS = {
    "Auto (mace-models default)": {"loader_name": None},
    "MACE-MP (small)": {"loader_name": "mace_mp_small"},
    "MACE-MP (medium)": {"loader_name": "mace_mp_medium"},
    "MACE-OFF23 (medium)": {"loader_name": "mace_off23_medium"},
}


@lru_cache(maxsize=16)
def get_calculator(model_family: str, variant: str):
    """
    Returns an ASE calculator for the selected family/variant.
    Lazy-imports heavy packages so the UI can render even if they’re missing.
    """
    family = (model_family or "CHGNet").strip()
    var = (variant or "CHGNet v0.4 (default)").strip()

    if family == "MACE":
        # Try mace-torch first (calculator class)
        try:
            from mace.calculators import MACECalculator  # pip: mace-torch
            have_mace = True
        except Exception as e:
            have_mace = False
            last_exc = e

        # Try mace-models for easy pretrained access
        have_mace_models = False
        if have_mace:
            try:
                import mace_models  # pip: mace-models
                have_mace_models = True
            except Exception:
                pass

        if not have_mace:
            raise RuntimeError(
                "MACE not installed. Add to requirements: 'mace-torch' (and optionally 'mace-models').\n"
                f"Import error: {last_exc}"
            )

        # Preferred path: mace-models
        if have_mace_models:
            try:
                loader_name = _MACE_MODELS.get(var, {}).get("loader_name")
                import mace_models
                model = mace_models.load() if loader_name is None else mace_models.load(loader_name)
                calc = model.get_calculator()
                return calc
            except Exception as e:
                # fall back to raw MACECalculator below
                fallback_err = e

        # Fallback path: raw MACECalculator reading model path from env
        import os
        mp = os.environ.get("MACE_MODEL_PATH", "").strip()
        if not mp:
            msg = "Set env var MACE_MODEL_PATH to a .model checkpoint or install 'mace-models'."
            if 'fallback_err' in locals():
                msg += f"\n(While trying mace-models, got: {fallback_err})"
            raise RuntimeError(msg)
        return MACECalculator(model_paths=mp, device="cpu")

    # Default family: CHGNet
    try:
        from chgnet.model import CHGNetCalculator  # pip: chgnet
    except Exception as e:
        raise RuntimeError(
            "CHGNet not installed. Add to requirements: 'chgnet'.\n"
            f"Import error: {e}"
        )

    cfg = _CHGNET.get(var, {})
    return CHGNetCalculator(**cfg)
