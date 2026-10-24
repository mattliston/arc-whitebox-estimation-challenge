"""Prespecified small input-only certificate. No network/target files."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from dirichlet_gram_observer_v1 import (
    covariance_coordinates, gate_hermite5, transport, zero_mean_gate,
)
from gaussian_shadow_disagreement_v1 import gaussian_step

HERE = Path(__file__).resolve().parent
OUT = HERE / 'dirichlet_gram_small_certificate_v1.json'
assert not OUT.exists()
rng = np.random.default_rng(930079)
records = []
with threadpool_limits(limits=2):
    for n in (9, 17, 31):
        for trial in range(3):
            a = rng.normal(size=(n, n)) / np.sqrt(n)
            cov = a @ a.T + np.eye(n) * 0.25
            mu = rng.normal(size=n)
            cov, sigma, alpha, rho, _ = covariance_coordinates(mu, cov)
            q = gate_hermite5(alpha, rho)
            epsilon = 1e-5
            direction = np.outer(sigma, sigma)
            np.fill_diagonal(direction, 0.0)
            plus = gaussian_step(mu, cov + epsilon * direction)[1]
            minus = gaussian_step(mu, cov - epsilon * direction)[1]
            mask = ~np.eye(n, dtype=bool)
            finite_difference = (plus[mask] - minus[mask]) / (2 * epsilon * direction[mask])
            error = finite_difference - q[mask]
            w = rng.normal(size=(n, n)) / np.sqrt(n)
            scale = np.exp(rng.normal(size=n) * 0.25)
            direct = transport(cov, w, product_dtype=np.float64)
            rescaled = transport(cov * np.outer(scale, scale), w / scale[:, None], product_dtype=np.float64)
            explicit = w.T @ cov @ w
            records.append(dict(
                n=n, trial=trial,
                derivative_max_abs=float(np.max(np.abs(error))),
                derivative_relative_l2=float(np.linalg.norm(error) / np.linalg.norm(q[mask])),
                gate_diagonal_max=float(np.max(np.abs(np.diag(q) - ndtr(alpha)))),
                symmetry_max=float(np.max(np.abs(q - q.T))),
                smallest_gate_eigenvalue=float(np.linalg.eigvalsh(q)[0]),
                transport_max_abs=float(np.max(np.abs(direct - explicit))),
                hidden_scale_max_abs=float(np.max(np.abs(direct - rescaled))),
            ))

correlations = np.array([0.0, 0.5, -0.5, np.sqrt(0.5), -np.sqrt(0.5)])
exact_values = np.array([0.25, 1/3, 1/6, 3/8, 1/8])
arcsin_error = float(np.max(np.abs(zero_mean_gate(correlations) - exact_values)))
passed = arcsin_error <= 1e-10 and all(
    r['derivative_max_abs'] <= 1e-7 and r['derivative_relative_l2'] <= 1e-7
    and r['gate_diagonal_max'] <= 1e-10 and r['symmetry_max'] <= 1e-10
    and r['smallest_gate_eigenvalue'] >= -1e-10
    and r['transport_max_abs'] <= 1e-10 and r['hidden_scale_max_abs'] <= 1e-10
    for r in records
)
result = dict(
    passed=passed, records=records, arcsin_max_abs=arcsin_error,
    target_access=False, network_array_access=False,
    seed=930079, source_sha256=hashlib.sha256((HERE / 'dirichlet_gram_observer_v1.py').read_bytes()).hexdigest(),
    certificate_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Small synthetic law/implementation checks; not deep Jacobian accuracy, wide PSD, native metering or board score.',
)
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
assert passed
