"""TAG opus-physicist: the truth loaders of pcard.py without torch (for numpy-only analyses)."""
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent


def truth(rank):
    return np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)


def pu(rank, tag='P32M'):
    return np.load(P2 / f'public_g19_attribution_v1/mps_truth_rank{rank:04d}_{tag}.npz')
