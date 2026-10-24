"""Frozen tiny numerical qualification; no Phase 2 input access."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import mpmath as mp
import numpy as np
from scipy.special import ndtr

import paired_mehler_probe_core_v1 as core


def mp_values(s, mu, sigma):
    with mp.workdps(100):
        s, mu, sigma = map(mp.mpf, (str(s), str(mu), str(sigma)))
        a, z = mu/sigma, (s-mu)/sigma
        phi = lambda v: mp.exp(-v*v/2)/mp.sqrt(2*mp.pi)
        cdf = lambda v: mp.erfc(-v/mp.sqrt(2))/2
        if s <= 0:
            c = a*cdf(a)+phi(a)
            r = cdf(z)/phi(z)
            return float(-c*r), float(-c*(1+z*r)/sigma)
        c = phi(a)-a*cdf(-a)
        r = cdf(-z)/phi(z)
        return float(-1+c*r), float(c*(z*r-1)/sigma)


def main():
    output = Path(sys.argv[1])
    if output.exists():
        raise SystemExit("refusing existing receipt")
    errors = [0., 0.]
    points = 0
    for alpha in (-64, -16, -8, -2, 0, 2, 8, 16, 64):
        for z in (-256, -64, -16, -8, -2, 0, 2, 8, 16, 64, 256):
            for sigma in (.125, 1., 3.):
                s, mu = (alpha+z)*sigma, alpha*sigma
                g, gp = core.stein(s, mu, sigma)
                rg, rgp = mp_values(s, mu, sigma)
                errors[0] = max(errors[0], abs(float(g)-rg))
                errors[1] = max(errors[1], abs(sigma*(float(gp)-rgp)))
                points += 1
    if max(errors) > 2e-10:
        raise AssertionError(("Stein grid", errors))
    rng = np.random.default_rng(937109)
    depth, rows, n = 3, 5, 3
    weights = rng.normal(size=(depth, n, n))*0.45
    mu = rng.normal(size=(depth, n))*0.2
    sigma = np.full((depth, n), 0.9)
    x0, z0 = rng.normal(size=(2, rows, n))
    t = rng.uniform(0.05, 0.95, size=rows)
    probes = rng.normal(size=(2, rows, n))
    got = core.evaluate(weights, mu, sigma, x0, z0, t, probes, detailed=True)
    jx = jy = jb = None
    mx = my = None
    max_jvp_error = 0.
    slopes = ndtr(mu/sigma)
    for layer, h in enumerate(got["histories"]):
        w = weights[layer]
        if layer == 0:
            jx = np.broadcast_to(w.T, (rows, n, n)).copy()
            jy, jb = jx.copy(), w.T.copy()
        else:
            jx = np.einsum("ij,bjk->bik", w.T, mx[:, :, None]*jx)
            jy = np.einsum("ij,bjk->bik", w.T, my[:, :, None]*jy)
            jb = w.T@(slopes[layer-1][:, None]*jb)
        expected = (np.einsum("boi,rbi->rbo", jx, probes),
                    np.einsum("boi,rbi->rbo", jy, probes),
                    probes@jb.T)
        for key, exact in zip(("tx", "ty", "tc"), expected):
            max_jvp_error = max(max_jvp_error, float(np.max(np.abs(h[key]-exact))))
        mx, my = h["ax"] > 0, h["ay"] > 0
    if max_jvp_error > 1e-11:
        raise AssertionError(("JVP", max_jvp_error))
    matrices = np.zeros((rows, n, n, n))  # row, output, input, input
    def value(e):
        pair = np.stack((e, np.zeros_like(e)))
        return core.evaluate(weights, mu, sigma, x0, z0, t, pair)["mean_difference"]
    for row in range(rows):
        diagonal_values = []
        for i in range(n):
            e = np.zeros((rows, n)); e[row, i] = 1
            diagonal_values.append(value(e))
            matrices[row, :, i, i] = diagonal_values[-1]
        for i in range(n):
            for j in range(i):
                e = np.zeros((rows, n)); e[row, i] = e[row, j] = 1
                off = (value(e)-diagonal_values[i]-diagonal_values[j])/2
                matrices[row, :, i, j] = matrices[row, :, j, i] = off
    def quadratic(e):
        return np.einsum("bi,boij,bj->o", e, matrices, e)
    expected_difference = quadratic(probes[0])-quadratic(probes[1])
    quadratic_error = float(np.max(np.abs(expected_difference-got["mean_difference"])))
    if quadratic_error > 1e-10:
        raise AssertionError(("quadratic", quadratic_error))
    affine = core.evaluate(weights[:1], mu[:1], sigma[:1], x0, z0, t, probes)
    if affine["statistic"] != 0 or np.any(affine["mean_difference"]):
        raise AssertionError("affine matching control not exactly zero")
    result = dict(passed=True, scope="tiny FP64 qualification only; no Phase2 data",
                  finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  core_sha256=sha256(Path(core.__file__).read_bytes()).hexdigest(),
                  script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  stein_grid_points=points, max_abs_g_error=errors[0],
                  max_abs_sigma_gp_error=errors[1], max_abs_jvp_error=max_jvp_error,
                  max_abs_quadratic_error=quadratic_error,
                  affine_difference_exact_zero=True,
                  tiny_analytic_expected_Y=float(rows/n*2*np.sum(matrices*matrices)),
                  tiny_observed_Y=got["statistic"])
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
