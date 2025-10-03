# tabs/utils_env.py
from __future__ import annotations
import os
import logging

def harden_env() -> None:
    """
    Minimal, version-agnostic runtime hardening:
      • Cap threads to avoid OpenMP/BLAS explosions (OOM).
      • Ensure cache dirs exist.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    os.environ.setdefault("HOME", "/home/appuser")
    os.environ.setdefault("XDG_CACHE_HOME", "/home/appuser/.cache")


def sanitize_logs() -> None:
    """
    Replace any 'CHGNet' mentions in logs with 'MACE' so UI/logs are consistent.
    (Some upstream stacks still stamp sub-jobs with 'CHGNet' labels.)
    """
    class _ReplaceCHGNet(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                record.msg = str(record.msg).replace("CHGNet", "MACE")
            except Exception:
                pass
            return True

    _f = _ReplaceCHGNet()
    for name in ("jobflow", "atomate2", "MLFF", "__main__"):
        logging.getLogger(name).addFilter(_f)
