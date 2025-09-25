# core/model.py
from __future__ import annotations
import functools
from mace.calculators import mace_mp
from ase.calculators.emt import EMT

@functools.lru_cache(maxsize=2)
def get_calculator(model_family: str = "MACE", variant: str = "default"):
    """
    Returns a cached ASE calculator.
    - Only MACE is supported now. 'variant' kept for future switch (e.g., medium/large).
    """
    _ = variant  # currently unused
    return mace_mp()  # MACE-MP default
