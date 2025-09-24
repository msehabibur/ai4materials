from __future__ import annotations
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def msd_and_plot(positions: np.ndarray, dt_fs: float) -> bytes:
    """
    positions: (n_steps, n_atoms, 3) Å
    dt_fs: timestep fs
    Returns PNG bytes for MSD vs Time plot.
    """
    r0 = positions[0]
    dr = positions - r0
    msd = (dr**2).sum(axis=2).mean(axis=1)
    t_ps = np.arange(len(msd)) * dt_fs / 1000.0

    fig = plt.figure()
    plt.plot(t_ps, msd)
    plt.xlabel("Time (ps)")
    plt.ylabel("MSD (Å$^2$)")
    plt.title("Mean Squared Displacement")
    buf = io.BytesIO()
    fig.savefig(buf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
