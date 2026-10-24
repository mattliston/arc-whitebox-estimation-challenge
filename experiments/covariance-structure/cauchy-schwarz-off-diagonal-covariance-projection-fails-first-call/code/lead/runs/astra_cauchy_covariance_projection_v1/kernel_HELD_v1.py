"""HELD synthetic kernel only; no candidate/model import. Fixed numerical rule."""
def project_pair_covariance(c32, diagonal64, fnp):
    n = int(c32.shape[0])
    assert tuple(c32.shape) == (n, n) and tuple(diagonal64.shape) == (n,)
    assert str(c32.dtype) == 'float32' and str(diagonal64.dtype) == 'float64'
    assert bool(fnp.all(fnp.isfinite(diagonal64)))
    assert bool(fnp.all(diagonal64 >= 0.0))
    assert bool(fnp.all(fnp.isfinite(c32)))
    # One predecessor compensates either nearest conversion's upward rounding.
    # Zero remains zero; overflow to +inf steps to finite max, still a lower bound.
    scale = fnp.nextafter(fnp.sqrt(diagonal64).astype(fnp.float32), 0.0)
    bound = fnp.nextafter(fnp.outer(scale, scale), 0.0)
    fnp.fill_diagonal(bound, fnp.inf)
    out = fnp.clip(c32, -bound, bound)
    changed = int(fnp.sum(fnp.not_equal(out, c32)))
    return out, changed
