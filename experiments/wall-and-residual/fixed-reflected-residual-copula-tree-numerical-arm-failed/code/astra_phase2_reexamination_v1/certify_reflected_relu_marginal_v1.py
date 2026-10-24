"""Inert-by-default synthetic marginal algebra; no network or target inputs."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from scipy.integrate import quad


def density(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def moments(alpha):
    t = abs(alpha)
    q = 0.5 * math.erfc(t / math.sqrt(2))
    r = density(t) - t * q
    if alpha >= 0:
        m = alpha + r
        v = 1 - q - t * r - r * r
        k = (t*t-1)*r + 2*t*q + 3*r*q + 3*t*r*r + 2*r*r*r
    else:
        m = r
        v = q - t * r - r * r
        k = (t*t+2)*r - t*q - 3*r*q + 3*t*r*r + 2*r*r*r
    if not (math.isfinite(m + v + k) and m > 0 and v > 0 and k > 0):
        raise ArithmeticError((alpha, m, v, k))
    return m, v, k


def integrated_moments(alpha):
    t = abs(alpha)
    tails = [quad(lambda u: u**j * density(t+u), 0.0, math.inf,
                  epsabs=1e-100, epsrel=1e-12, limit=200)[0] for j in (1, 2, 3)]
    r1, r2, r3 = tails
    if alpha < 0:
        return r1, r2-r1*r1, r3-3*r1*r2+2*r1*r1*r1
    lost_second = 2*t*r1 + r2
    m = alpha + r1
    v = 1-lost_second-r1*r1
    k = (3*t*t*r1+3*t*r2+r3) - 3*r1*(1-lost_second) + 2*r1*r1*r1
    return m, v, k


def skew(alpha):
    _, v, k = moments(alpha)
    return k / (v * math.sqrt(v))


def invert(target):
    lo, hi = -8.0, 12.0
    if not (skew(hi) <= target <= skew(lo)):
        raise ValueError("Outside frozen synthetic inversion bracket")
    for _ in range(64):
        mid = (lo+hi)/2
        if skew(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def execute(out):
    if out.exists():
        raise FileExistsError(out)
    rows = []
    for alpha in (-8.0, -6.0, -3.0, -1.0, 0.0, 0.75, 2.0, 4.0, 8.0, 12.0):
        actual = moments(alpha)
        expected = integrated_moments(alpha)
        errors = [abs(x-y)/abs(y) for x, y in zip(actual, expected)]
        m, v, k = actual
        q = 0.5 * math.erfc(alpha / math.sqrt(2))
        # Stable centered spelling of m2^2-m1*m3.
        numerator = v * (v-m*m) - m*k
        derivative = 3*q*numerator/(v*v*math.sqrt(v))
        rows.append(dict(alpha=alpha, moments=actual, relative_errors=errors,
                         skew=skew(alpha), skew_derivative=derivative))
    inversions = []
    for alpha in (-6.0, -3.0, -0.5, 0.0, 1.0, 4.0, 8.0):
        recovered = invert(skew(alpha))
        inversions.append(dict(alpha=alpha, recovered=recovered,
                               absolute_error=abs(recovered-alpha)))
    worst = max(x for row in rows for x in row["relative_errors"])
    inversion_error = max(row["absolute_error"] for row in inversions)
    decreasing = all(row["skew_derivative"] < 0 for row in rows)
    result = dict(scope="synthetic scalar moments and finite-bracket inversion only",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  finished_utc=datetime.now(timezone.utc).isoformat(), rows=rows,
                  inversions=inversions, worst_moment_relative_error=worst,
                  worst_alpha_recovery_error=inversion_error,
                  sampled_derivatives_strictly_negative=decreasing,
                  passed=bool(worst <= 1e-8 and inversion_error <= 1e-10 and decreasing))
    with out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "inversions")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", type=Path,
                   default=Path(__file__).with_name("reflected_relu_marginal_certificate_v1.json"))
    args = p.parse_args()
    if args.execute:
        execute(args.output)
    else:
        print("INERT: synthetic scalar algebra only; --execute required.")
