"""Fixed Gaussian6 forecast plus terminal frozen-gate cubic response.

No learned payload, CP bank, NumPy, external state, or numerical import work.
Gaussian formulas follow parent 409f903e gaussian_raw/derivatives; the new
birth and terminal-only closure are specified in the companion PROTOCOL.md.
All dense products are ordinary public FP32 matmuls, not packed/FMM kernels.
"""
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator


def _finite(x, name):
    if not bool(fnp.all(fnp.isfinite(x))):
        raise ValueError("Nonfinite " + name)


def _mm(a, b, ledger):
    assert a.dtype == fnp.float32 and b.dtype == fnp.float32
    ledger.append((int(a.shape[0]), int(a.shape[1]), int(b.shape[1])))
    return a @ b


def gaussian_raw(mu, variance):
    """FP64 scalar Gaussian moments, with the inherited 1e-10 input floor."""
    v = fnp.maximum(variance, 1e-10)
    s = fnp.sqrt(v)
    a = mu / s
    pdf = fnp.exp(-a * a / 2) / 2.5066282746310002
    g = flops.stats.norm.cdf(a)
    m = s * pdf + mu * g
    r2 = (mu * mu + v) * g + mu * s * pdf
    r3 = mu * (mu * mu + 3 * v) * g + (mu * mu + 2 * v) * s * pdf
    post_v = r2 - m * m
    k3 = r3 - 3 * m * r2 + 2 * m * m * m
    return dict(v=v, s=s, a=a, pdf=pdf, g=g, m=m, post_v=post_v,
                k3=k3, d=pdf / s, e=-a * pdf / v)


def gaussian_covariance(sigma32, scalar):
    """FP64 positive-coefficient order-six Mehler map; FP32 working output.

No rho clipping. Diagonal is replaced by the FP64 marginal variance before
the final FP32 cast. This polynomial is not the exact bivariate Gaussian law.
"""
    s, a, pdf, g = (scalar[k] for k in ('s', 'a', 'pdf', 'g'))
    rho = sigma32.astype(fnp.float64) / (s[:, None] * s[None, :])
    fnp.fill_diagonal(rho, 0.0)
    h0, h1 = fnp.ones_like(a), a
    coefficients = [g]
    for r in range(2, 7):
        if r == 2:
            h = h0
        elif r == 3:
            h = h1
        else:
            h = a * h1 - (r - 3) * h0
            h0, h1 = h1, h
        coefficients.append(((-1) ** (r - 2)) * h * pdf)
    rr = fnp.ones_like(rho)
    answer = fnp.zeros_like(rho)
    for order, coef in enumerate(coefficients, 1):
        rr = rr * rho / order
        answer = answer + rr * (coef[:, None] * coef[None, :])
    answer = answer * (s[:, None] * s[None, :])
    fnp.fill_diagonal(answer, scalar['post_v'])
    _finite(answer, 'Gaussian covariance')
    return answer.astype(fnp.float32)


def make_birth(sigma32, scalar):
    """Column-i source B[:,i]; gamma uses the ACTUAL stored FP32 B diagonal."""
    base = (scalar['g'][:, None] * sigma32.astype(fnp.float64)).astype(fnp.float32)
    diag = fnp.diag(base).astype(fnp.float64)
    gamma = scalar['k3'] - 3 * scalar['d'] * diag * diag
    _finite(base, 'birth B')
    _finite(gamma, 'birth gamma')
    return base, scalar['d'], gamma


def birth_diagonal(j32, birth, ledger):
    """Exact polynomial contraction in real arithmetic of this stored source."""
    base, d, gamma = birth
    u = _mm(j32, base, ledger).astype(fnp.float64)
    j = j32.astype(fnp.float64)
    value = fnp.sum(3 * (j * u * u) * d[None, :]
                    + (j * j * j) * gamma[None, :], axis=1)
    _finite(value, 'transported birth diagonal')
    return value


def forecast(weights, with_births):
    """One fixed causal Gaussian forecast, identical in both acquisition roles."""
    depth, n = len(weights), int(weights[0].shape[0])
    assert depth >= 2
    ledger, means, births, gates = [], [], [], []
    m = fnp.zeros(n, dtype=fnp.float64)
    covariance = None
    floors = 0
    for layer, w in enumerate(weights):
        assert isinstance(w, fnp.ndarray) and w.dtype == fnp.float32
        assert w.shape == (n, n)
        w64 = w.astype(fnp.float64)
        mu = w64.T @ m
        if layer == 0:
            sigma = _mm(w.T, w, ledger)
        else:
            cw = _mm(covariance, w, ledger)
            if layer != depth - 1:
                sigma = _mm(w.T, cw, ledger)
        if layer == depth - 1:
            variance = fnp.sum(w64 * cw.astype(fnp.float64), axis=0)
        else:
            sigma = (sigma + sigma.T) * .5
            _finite(sigma, 'pre-covariance')
            variance = fnp.diag(sigma).astype(fnp.float64)
        _finite(variance, 'pre-variance')
        floors += int(fnp.sum(variance < 1e-10))
        scalar = gaussian_raw(mu, variance)
        for name, value in scalar.items():
            _finite(value, name)
        if not bool(fnp.all(scalar['post_v'] >= 0)):
            raise ValueError('Negative Gaussian marginal variance; no repair')
        m = scalar['m']
        means.append(m)
        if layer != depth - 1:
            covariance = gaussian_covariance(sigma, scalar)
            if with_births:
                births.append(make_birth(sigma, scalar))
                gates.append(scalar['g'].astype(fnp.float32))
    assert len(ledger) == 2 * depth - 2
    return dict(means=means, births=births, gates=gates, terminal=scalar,
                ledger=ledger, pre_variance_floor_count=floors)


def terminal_skew(weights, state):
    depth = len(weights)
    j = weights[-1].T
    total = fnp.zeros(int(j.shape[0]), dtype=fnp.float64)
    before = len(state['ledger'])
    for layer in range(depth - 2, -1, -1):
        total = total + birth_diagonal(j, state['births'][layer], state['ledger'])
        if layer:
            j = _mm(j * state['gates'][layer][None, :],
                    weights[layer].T, state['ledger'])
            _finite(j, 'suffix')
    assert len(state['ledger']) - before == 2 * depth - 3
    return total


def _predict(weights, corrected):
    state = forecast(weights, corrected)
    means = state['means']
    if corrected:
        k3 = terminal_skew(weights, state)
        value = means[-1] + state['terminal']['e'] * k3 / 6
        _finite(value, 'unclipped terminal mean')
        means = means[:-1] + [fnp.maximum(value, 0.0)]
    output = fnp.stack(means, axis=0).astype(fnp.float32)
    _finite(output, 'output')
    return output, dict(fp32_matrix_products=len(state['ledger']),
                        shapes=state['ledger'],
                        pre_variance_floor_count=state['pre_variance_floor_count'],
                        terminal_only_correction=bool(corrected))


class GaussianEstimator(BaseEstimator):
    """Fixed independent control; no birth extraction or suffix work charged."""
    def predict(self, mlp, budget):
        output, self.last_diagnostics = _predict(mlp.weights, False)
        return output


class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        output, self.last_diagnostics = _predict(mlp.weights, True)
        return output
