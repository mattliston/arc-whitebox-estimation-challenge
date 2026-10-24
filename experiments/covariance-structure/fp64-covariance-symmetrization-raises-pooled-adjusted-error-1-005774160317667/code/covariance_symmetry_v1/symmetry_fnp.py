def _covsym_full(x):
    """Orthogonal projection onto symmetric matrices; no PSD modification."""
    if len(x.shape) != 2 or x.shape[0] != x.shape[1] or x.dtype != fnp.float64:
        raise ValueError('covariance symmetry requires a square FP64 matrix')
    return (x + x.T) * .5
