"""Exact-rational checks of a source-only spherical Stein cost obstruction.

No network, data, target, random samples, fit, or estimator is executed.
This checks the quadratic-harmonic instance and the minimization constants;
the general Bochner identity still needs its analytic proof.
"""

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def quadratic_check(n):
    # Radius one; u=x^T A x, tr(A)=0 and tr(A^2)=2 (e.g. diag(1,-1,...)).
    m = n - 1
    trace_a2 = F(2)
    eu2 = 2 * trace_a2 / (n * (n + 2))
    exa2x = trace_a2 / n
    # Trace-free intrinsic Hessian B0=2 P A P + (2/m) u P.
    direct_b0_energy = 4 * trace_a2 - 8 * exa2x + 4 * F(m - 1, m) * eu2
    lam = 2 * n
    bochner_b0_energy = F(m - 1, m) * lam * (lam - m) * eu2
    assert direct_b0_energy == bochner_b0_energy
    control_energy = F(lam * lam, m * m) * eu2
    noise_energy = F(2, m * m) * direct_b0_energy
    rho = F(2 * (m - 1), m) * (1 - F(m, lam))
    assert noise_energy / control_energy == rho
    assert rho == 1 - F(2, m * (m + 1))
    return {
        "n": n,
        "eu2": str(eu2),
        "trace_free_hessian_energy": str(direct_b0_energy),
        "gaussian_noise_control_ratio": str(rho),
        "quadratic_identity_exact": True,
    }


def main():
    output = Path(__file__).with_name("spherical_stein_constants_certificate_v1.json")
    if output.exists():
        raise RuntimeError("Refusing to overwrite a versioned receipt")
    rows = [quadratic_check(n) for n in (3, 4, 8, 16, 64, 1024)]
    n = 1024
    m = n - 1
    rho_min = 1 - F(2, m * (m + 1))
    # For l>=2, rho_l>=rho_min. For every fixed q>=1,
    # (46+31q)*rho/[16(q+rho)] increases in rho.
    # At rho_min the q-derivative has sign(31*rho_min-46)<0.
    # Therefore the infimum over all q>=1 and l>=2 is the q->infinity
    # limit 31*rho_min/16, an optimistic bound (finite budgets limit q).
    assert 31 * rho_min - 46 < 0
    floor_factor = F(31, 16) * rho_min
    assert floor_factor > 1
    for ell in (2, 3, 4, 8, 16, 64, 1024):
        lam = ell * (ell + n - 2)
        rho = F(2 * (m - 1), m) * (1 - F(m, lam))
        assert rho >= rho_min
        for q in (1, 2, 4, 16, 256, 65536):
            factor = F(46 + 31 * q, 16) * rho / (q + rho)
            assert factor > floor_factor
    haar_rademacher_rho = F(n, n + 2) * rho_min
    assert 31 * haar_rademacher_rho - 46 < 0
    haar_floor_factor = F(31, 16) * haar_rademacher_rho
    assert haar_floor_factor > 1
    # This is a declared classical-dense circuit yardstick, not a native bill.
    plain_path_flops = 2 * 16 * n ** 2
    budget = 2 ** 41
    required_nonlinear_variance = F(1, 10 ** 9) * budget / (floor_factor * plain_path_flops)
    receipt = {
        "passed": True,
        "scope": "exact rational constants, not a network variance or board score",
        "network_or_target_access": False,
        "general_bochner_proof_checked_by_code": False,
        "quadratic_checks": rows,
        "n": n,
        "plain_path_flops": plain_path_flops,
        "literal_stein_matrix_products": "46 + 31*q",
        "gaussian_nonlinear_variance_cost_infimum_exact": str(floor_factor),
        "gaussian_nonlinear_variance_cost_infimum": float(floor_factor),
        "haar_orientation_rademacher_infimum": float(haar_floor_factor),
        "fixed_or_adapted_rademacher_basis_bound_claimed": False,
        "necessary_nonlinear_variance_for_adjusted_1e_9": float(required_nonlinear_variance),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
