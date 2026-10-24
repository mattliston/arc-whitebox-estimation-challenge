"""Root-only guarded toy certificate for fixed cfcov; no network or truth reads.

Prepared source-only. Run with --run-guarded inside the root's authorized guard.
Independent raw-joint autograd derivatives verify the connected subtraction.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-guarded', action='store_true')
    args = parser.parse_args()
    assert args.run_guarded, 'Root must use its authorized guard process'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import torch as t
    t.set_num_threads(2)
    t.set_num_interop_threads(1)
    path = Path(__file__).resolve().parent / 'cutsim_cfcov_v1/pbase_cutsim_v1.py'
    spec = importlib.util.spec_from_file_location('astra_cfcov_cert_module', path)
    pb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pb)
    dt = t.float64
    s = t.tensor([.7, 1.3, .9, 1.1], dtype=dt)
    mu = s * t.tensor([-1.2, -.3, .6, 1.7], dtype=dt)
    rho = t.tensor([[1., .12, -.08, .03], [.12, 1., .21, -.07],
                    [-.08, .21, 1., .11], [.03, -.07, .11, 1.]], dtype=dt)
    C = rho * t.outer(s, s)

    def gaussian(mean, covariance):
        sigma = t.diagonal(covariance).sqrt()
        x = mean / sigma
        ph = t.exp(-x * x / 2) / math.sqrt(2 * math.pi)
        gate = .5 * t.erfc(-x / math.sqrt(2))
        m = sigma * (x * gate + ph)
        hs = [t.ones_like(x), x]
        for k in range(2, 5): hs.append(x * hs[-1] - (k - 1) * hs[-2])
        fs = [None, sigma * gate]
        fs += [sigma * (-1) ** (k - 2) * hs[k - 2] * ph for k in range(2, 7)]
        rr = covariance / t.outer(sigma, sigma)
        V = sum(rr ** k / math.factorial(k) * t.outer(fs[k], fs[k]) for k in range(1, 7))
        return m, V

    def auxiliary(mean, covariance, beta):
        sigma = t.diagonal(covariance).sqrt()
        x = mean / sigma
        ph = t.exp(-x * x / 2) / math.sqrt(2 * math.pi)
        gate = .5 * t.erfc(-x / math.sqrt(2))
        u = covariance @ mean / t.sqrt(mean @ covariance @ mean)
        return dict(sigma=sigma, alpha=x, pdf=ph, gate=gate,
                    gaussian_mean=sigma * (x * gate + ph),
                    pre_k3=3 * beta * sigma ** 2 * u)

    def joint(mean, covariance):
        m, V = gaussian(mean, covariance)
        return V + t.outer(m, m)

    def first(fn, zero):
        return t.autograd.functional.jacobian(fn, zero)

    def second(fn, zero):
        return t.autograd.functional.hessian(fn, zero)

    def reference_linear(c, B, q):
        zero = t.tensor(0., dtype=dt, requires_grad=True)
        m0 = gaussian(mu, C)[0]
        m2 = t.stack([second(lambda z: gaussian(mu + z * c, C + z * B)[0][i], zero)
                      for i in range(4)])
        H2 = t.stack([t.stack([second(lambda z: joint(mu + z * c, C + z * B)[i, j], zero)
                               for j in range(4)]) for i in range(4)])
        compensation = -q * t.outer(c, c)
        mdot = t.stack([first(lambda z: gaussian(mu, C + z * compensation)[0][i], zero)
                        for i in range(4)])
        Hdot = t.stack([t.stack([first(lambda z: joint(mu, C + z * compensation)[i, j], zero)
                                 for j in range(4)]) for i in range(4)])
        return (.5 * q * (H2 - t.outer(m2, m0) - t.outer(m0, m2))
                + Hdot - t.outer(mdot, m0) - t.outer(m0, mdot))

    checks = []
    for r, beta in ((.06, .065), (.18, .13), (.3, .2), (.06, 0.)):
        aux = auxiliary(mu, C, beta)
        model = pb._cfcov_setup(t, mu, C, aux, r)
        assert model is not None
        q, c, B = model['q'], model['c'], model['B']
        assert abs(model['beta'] - beta) < 1e-14
        assert t.linalg.eigvalsh(B).min() > 0
        nodes, weights = model['nodes'], model['weights']
        for k in range(10):
            expected = 0.0
            if k % 2 == 0:
                expected = q ** (k // 2) * math.prod(range(1, k, 2))
            actual = sum(w * d ** k for w, d in zip(weights, nodes))
            assert abs(actual - expected) < 1e-13, (k, actual, expected)
        mean_shift = sum(w * c * d for w, d in zip(weights, nodes))
        input_cov = sum(w * ((1 + d) * B + d * d * t.outer(c, c))
                        for w, d in zip(weights, nodes))
        assert mean_shift.abs().max() < 1e-14
        assert t.allclose(input_cov, C, rtol=1e-13, atol=1e-14)
        for direction in (t.tensor([1., -.2, .3, .4], dtype=dt),
                          t.tensor([-.4, .7, -.1, .2], dtype=dt)):
            ac, bc, vc = float(direction @ c), float(direction @ B @ direction), float(direction @ C @ direction)
            k3 = sum(w * ((ac * d) ** 3 + 3 * ac * d * (1 + d) * bc)
                     for w, d in zip(weights, nodes))
            k4 = sum(w * ((ac * d) ** 4 + 6 * (ac * d) ** 2 * (1 + d) * bc
                           + 3 * ((1 + d) * bc) ** 2)
                     for w, d in zip(weights, nodes)) - 3 * vc * vc
            assert abs(k3 - 3 * q * ac * bc) < 1e-12
            assert abs(k4 - 3 * q * bc * bc) < 1e-12
        linear_ref = reference_linear(c, B, q)
        linear = pb._cfcov_linear(t, C, aux, q, c, B)
        linear_error = float((linear - linear_ref).abs().max())
        assert t.allclose(linear, linear_ref, rtol=1e-9, atol=2e-12), linear_error
        rows = [gaussian(mu + c * d, (1 + d) * B) for d in nodes]
        mbar = sum(w * m for w, (m, V) in zip(weights, rows))
        Hbar = sum(w * (V + t.outer(m, m)) for w, (m, V) in zip(weights, rows))
        reference = Hbar - t.outer(mbar, mbar) - gaussian(mu, C)[1] - linear_ref
        reference = t.diagonal_scatter(reference, t.zeros_like(s))
        got = pb._cfcov_delta(t, mu, C, aux, r)
        error = float((got - reference).abs().max())
        assert t.allclose(got, reference, rtol=1e-8, atol=3e-12), error
        assert t.equal(t.diagonal(got), t.zeros_like(s))
        assert t.allclose(got, got.T, rtol=0, atol=1e-14)
        assert t.isfinite(got).all()
        checks.append(dict(r=r, beta=beta, linear_error=linear_error,
                           remainder_error=error, max_remainder=float(got.abs().max())))
    aux = auxiliary(mu, C, .065)
    for r in (0., -.1, pb._CFCOV_RMAX, .5, float('nan'), float('inf')):
        assert pb._cfcov_delta(t, mu, C, aux, r) is None
    assert pb._cfcov_delta(t, mu, C, auxiliary(mu, C, 1.), .06) is None
    assert pb._cfcov_delta(t, t.zeros_like(mu), C, aux, .06) is None
    norms = []
    for epsilon in (.01, .005):
        small_aux = dict(aux, pre_k3=epsilon * aux['pre_k3'])
        delta = pb._cfcov_delta(t, mu, C, small_aux, .06 * epsilon)
        norms.append(float(t.linalg.vector_norm(delta)))
    ratio = norms[0] / norms[1]
    assert 3.85 < ratio < 4.15, ('first-order cancellation', ratio)
    far_mu = 20 * s
    identity = pb._cfcov_delta(t, far_mu, C, auxiliary(far_mu, C, .065), .06)
    identity_max = float(identity.abs().max())
    assert identity_max < 3e-12, identity_max
    print(json.dumps(dict(ok=True, source=str(path), checks=checks,
                          small_strength_norm_ratio=ratio,
                          identity_limit_max=identity_max), indent=2))


if __name__ == '__main__':
    main()
