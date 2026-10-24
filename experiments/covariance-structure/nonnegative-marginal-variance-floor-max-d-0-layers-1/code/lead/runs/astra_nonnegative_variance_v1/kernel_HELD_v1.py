"""One fixed marginal-variance projection; synthetic/source-only until registered."""
def project_diagonal(diagonal64, fnp):
    assert len(diagonal64.shape) == 1 and str(diagonal64.dtype) == 'float64'
    assert bool(fnp.all(fnp.isfinite(diagonal64)))
    answer = fnp.maximum(diagonal64, 0.0)
    changed = int(fnp.sum(fnp.not_equal(answer, diagonal64)))
    return answer, changed
