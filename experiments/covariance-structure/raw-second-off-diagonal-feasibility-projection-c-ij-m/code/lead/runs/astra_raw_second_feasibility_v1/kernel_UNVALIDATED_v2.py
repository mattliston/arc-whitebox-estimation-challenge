"""HELD isolated same-input raw-second projection; no model integration or grant.
Projected bound is in the native float32 covariance representation.
Original float64 diagonal is restored by the existing caller after this hook.
"""

def project_raw_second(cov32, mean64, delta64, fnp):
    assert str(cov32.dtype) == 'float32'
    assert str(mean64.dtype) == str(delta64.dtype) == 'float64'
    n = int(mean64.shape[0])
    assert tuple(cov32.shape) == (n,n) and tuple(mean64.shape) == tuple(delta64.shape) == (n,)
    # Duplicate only this O(n) sum, so every original caller operation retains its order.
    future_mean = fnp.asarray(mean64 + delta64, dtype=fnp.float32)
    lower = fnp.outer(-future_mean, future_mean)
    changed = fnp.less(cov32, lower)
    fnp.fill_diagonal(changed, False)
    answer = fnp.where(changed, lower, cov32)
    count = int(fnp.sum(changed))
    return answer, count
