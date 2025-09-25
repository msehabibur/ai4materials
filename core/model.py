from __future__ import annotations
from functools import lru_cache
from chgnet.model import CHGNetCalculator

@lru_cache(maxsize=1)
def get_chgnet_calculator():
    """Cached pretrained CHGNet ASE calculator."""
    return CHGNetCalculator()
