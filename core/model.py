from __future__ import annotations
from functools import lru_cache
from chgnet.model import CHGNetCalculator

# Map sidebar names to internal config. Extend here later if you add checkpoints.
_VARIANTS = {
    "CHGNet v0.4 (default)": {"kwargs": {}},
}

@lru_cache(maxsize=4)
def get_chgnet_calculator(variant: str = "CHGNet v0.4 (default)"):
    cfg = _VARIANTS.get(variant, _VARIANTS["CHGNet v0.4 (default)"])
    return CHGNetCalculator(**cfg["kwargs"])
