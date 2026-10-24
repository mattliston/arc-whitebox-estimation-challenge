import math
import resource

from common import *
from geometry import negative_modes, response


def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(x['num_threads'] == 1 for x in threadpool_info())
    verify()
    rng = np.random.default_rng(720035)
    rows = []
    for values in ([0., 0., 0., 0.], [0., 1., 2., 3.], [-3., -.7, .2, 4.], [-4., -3., -2., -1.]):
        U, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        C = (U * values) @ U.T
        skew = rng.normal(size=(4, 4)) * .01
        C = C + skew - skew.T
        lam, Q, d, meta = negative_modes(np, C)
        correction = (Q * d) @ Q.T
        expected = (U * np.maximum(-np.asarray(values), 0)) @ U.T
        relative = float(np.linalg.norm(correction - expected) / max(np.linalg.norm(expected), 1.))
        assert relative <= 1e-10
        W = rng.normal(size=(4, 5));factor = rng.uniform(.1, 1., size=5)
        ref = factor * np.diag(W.T @ correction @ W)
        got = response(np, Q, d, W, factor)
        response_error = float(np.linalg.norm(ref - got) / max(np.linalg.norm(ref), 1.))
        assert response_error <= 1e-10
        permutation = np.array([2, 0, 3, 1]);Cp = C[np.ix_(permutation, permutation)]
        _, Qp, dp, _ = negative_modes(np, Cp)
        assert np.allclose((Qp * dp) @ Qp.T, correction[np.ix_(permutation, permutation)], rtol=1e-10, atol=1e-12)
        V, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        _, Qv, dv, _ = negative_modes(np, V.T @ C @ V)
        assert np.allclose((Qv * dv) @ Qv.T, V.T @ correction @ V, rtol=1e-10, atol=1e-12)
        rows.append(dict(eigenvalues=values, correction_relative=relative, response_relative=response_error))
    C = np.diag(np.arange(-20., 4.))
    _, Q, d, meta = negative_modes(np, C)
    assert len(d) == 16 and np.array_equal(d, np.arange(20., 4., -1))
    assert abs(meta['retained_fraction'] - sum(x*x for x in range(5, 21))/sum(x*x for x in range(1, 21))) < 1e-15
    derivative_errors = []
    for mu, variance in ((-.7, .8), (0., .4), (.8, 1.2)):
        def mean(v):
            sigma = math.sqrt(v);alpha = mu/sigma
            return sigma*math.exp(-.5*alpha*alpha)/math.sqrt(2*math.pi) + mu*(1+math.erf(alpha/math.sqrt(2)))/2
        epsilon = 1e-5
        numerical = (mean(variance+epsilon)-mean(variance-epsilon))/(2*epsilon)
        exact = math.exp(-.5*mu*mu/variance)/math.sqrt(2*math.pi)/(2*math.sqrt(variance))
        error = abs(numerical-exact)/exact;assert error <= 1e-7;derivative_errors.append(error)
    write(HERE/'unit.json', dict(passed=True, rows=rows, derivative_relative=max(derivative_errors),
          rank16_selection_exact=True, targets_opened=False, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(passed=True, derivative_relative=max(derivative_errors))), flush=True)


if __name__ == '__main__':
    main()
