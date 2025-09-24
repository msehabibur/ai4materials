import numpy as np
from typing import List, Tuple
from pymatgen.core.structure import Structure

def _fractional_to_cartesian(struct: Structure, frac_coords: np.ndarray) -> np.ndarray:
    return frac_coords @ struct.lattice.matrix

def _unwrap(frac_coords_seq: np.ndarray) -> np.ndarray:
    unwrapped = frac_coords_seq.copy()
    for t in range(1, unwrapped.shape[0]):
        delta = unwrapped[t] - unwrapped[t-1]
        unwrapped[t] -= np.round(delta)
    return unwrapped

def compute_msd_and_diffusion(traj: List[Structure], dt_fs: float) -> Tuple[np.ndarray, np.ndarray, float]:
    n_frames = len(traj); n_atoms = len(traj[0])
    frac_seq = np.zeros((n_frames, n_atoms, 3))
    for i, s in enumerate(traj):
        frac_seq[i] = s.frac_coords
    frac_unwrapped = _unwrap(frac_seq)
    cart = np.zeros_like(frac_unwrapped)
    for i, s in enumerate(traj):
        cart[i] = _fractional_to_cartesian(s, frac_unwrapped[i])
    dr = cart - cart[0]
    msd = (dr**2).sum(axis=2).mean(axis=1)
    times_ps = (np.arange(n_frames) * dt_fs) / 1000.0
    half = n_frames // 2 if n_frames >= 10 else max(2, n_frames//2)
    slope, _ = np.polyfit(times_ps[half:], msd[half:], 1)
    D_A2_per_ps = slope / 6.0
    D_cm2_per_s = D_A2_per_ps * 1e-16
    return msd, times_ps, D_cm2_per_s