"""Tiny directed-rho and active-floor certificate, prepared for root review."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from radial_mehler import make_nonlinear
from pair_memory_action_physics_v1 import full_cubic_covariance_response as response_v1
from pair_memory_action_physics_v2 import offdiag, full_cubic_covariance_response


HERE = Path(__file__).resolve().parent
OUT = HERE / "pair_memory_response_certificate_v2.json"
SOURCE_NAMES = (
    Path(__file__).name, "pair_memory_action_physics_v1.py", "pair_memory_action_physics_v2.py",
    "radial_mehler.py", "k3_radial_numpy.py", "PAIR_MEMORY_RESPONSE_EXACTNESS_V2.md",
    "NONLINEAR_PAIR_MEMORY_INFORMATION_PROTOCOL_V1.md")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def hashes():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in SOURCE_NAMES}


def main():
    require(not OUT.exists(), f"Preserve existing certificate: {OUT}")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Launch at nice >=10")
    require("/anaconda3/" not in sys.executable, "Use the NumPy repository .venv interpreter")
    require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules), "No Torch")
    started = time.perf_counter()
    source_hashes = hashes()
    rng = np.random.default_rng(930119)
    nonlinear = make_nonlinear(6, 2, True, True)
    rows = []
    kinds = ["symmetric"]*6 + ["asymmetric", "asymmetric",
                                "sigma_floor_symmetric", "sigma_floor_asymmetric"]
    for index, kind in enumerate(kinds):
        n = 7
        factor = rng.standard_normal((n, n))/np.sqrt(n)
        c0 = factor@factor.T + np.eye(n) + np.ones((n, n))*.6
        covariance = c0/np.sqrt(np.outer(np.diag(c0), np.diag(c0)))
        scale = np.exp(rng.uniform(-1, 1, n))
        if "sigma_floor" in kind:
            scale[0] = 1e-6  # Fixed original variance 1e-12, below the 1e-10 sigma floor.
        covariance *= np.outer(scale, scale)
        if "asymmetric" in kind:
            draw = rng.standard_normal((n, n))
            covariance += 1e-4*(draw-draw.T)*np.outer(scale, scale)
        mu = rng.standard_normal(n)*scale
        k = rng.standard_normal((n, n))*.003*scale[:, None]**2*scale[None, :]
        variance = np.diag(covariance)
        k4 = .03*variance**2
        k22 = .03*(np.outer(variance, variance)+2*covariance*covariance)/3
        delta = offdiag(rng.standard_normal((n, n))*.003)
        base = nonlinear(mu, covariance, k, k4, k22, "covariance")
        aux = base[-1]
        response = full_cubic_covariance_response(
            delta, covariance, aux["sigma"], aux["alpha"], aux["pdf"], aux["gate"])
        legacy = response_v1(delta, covariance, aux["sigma"], aux["alpha"], aux["pdf"], aux["gate"])
        physical = delta*aux["sigma"][:, None]**2*aux["sigma"][None, :]
        actual = nonlinear(mu, covariance, k+physical, k4, k22, "covariance")[1]-base[1]
        half = nonlinear(mu, covariance, k+.5*physical, k4, k22, "covariance")[1]-base[1]
        denominator = float(np.linalg.norm(actual))
        require(np.isfinite(denominator) and denominator > 0, "Invalid fixture response denominator")
        relative = float(np.linalg.norm(response-actual)/denominator)
        linearity = float(np.linalg.norm(2*half-actual)/denominator)
        legacy_error = float(np.linalg.norm(legacy-actual)/denominator)
        legacy_difference = float(np.linalg.norm(legacy-response)/denominator)
        floor_count = int((variance < 1e-10).sum())
        asymmetry = float(np.linalg.norm(covariance-covariance.T)/np.linalg.norm(covariance))
        require(floor_count == (1 if "sigma_floor" in kind else 0), "Fixed floor fixture failed")
        if "asymmetric" in kind:
            require(asymmetry > 0, "Fixed asymmetric fixture lost its antisymmetric component")
        else:
            require(asymmetry == 0, "Symmetric fixture unexpectedly asymmetric")
        rows.append(dict(fixture=index, kind=kind, dimension=n,
                         full_response_relative_error=relative,
                         half_step_linearity_error=linearity,
                         legacy_v1_relative_error=legacy_error,
                         v1_vs_v2_relative_difference=legacy_difference,
                         covariance_relative_antisymmetry=asymmetry,
                         active_sigma_floor_count=floor_count))
    worst = max(max(row["full_response_relative_error"], row["half_step_linearity_error"])
                for row in rows)
    symmetric_difference = max(row["v1_vs_v2_relative_difference"] for row in rows
                               if "asymmetric" not in row["kind"])
    asymmetric_witness = max(row["legacy_v1_relative_error"] for row in rows
                             if "asymmetric" in row["kind"])
    passed = worst <= 2e-11 and symmetric_difference <= 2e-11 and asymmetric_witness > 1e-10
    require(hashes() == source_hashes, "A source changed during the tiny certificate")
    result = dict(schema="pair-memory-response-certificate-v2", passed=bool(passed),
                  fixtures=rows, rng_seed=930119, fixture_count=10, dimension=7,
                  worst_relative_error=worst,
                  worst_symmetric_v1_vs_v2_difference=symmetric_difference,
                  asymmetric_v1_error_witness=asymmetric_witness,
                  fixed_tolerance=2e-11, fixed_asymmetric_witness_floor=1e-10,
                  source_hashes=source_hashes, all_source_hashes_unchanged=True,
                  target_access=False, torch_imported=False, wide_execution=False,
                  wall_seconds=time.perf_counter()-started,
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification="Algebraically exact directed r=0..2 bank action with zero query diagonal; "
                                "not a bank/head Jacobian, feature fit, raw result or PSD repair")
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)
    require(passed, "Fixed v2 response certificate failed; preserve the receipt and do not sweep")


if __name__ == "__main__":
    main()
