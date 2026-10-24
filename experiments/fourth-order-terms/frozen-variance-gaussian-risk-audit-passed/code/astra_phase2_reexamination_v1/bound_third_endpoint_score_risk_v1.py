"""No-refit same-ridge risk bound using the computed normal residual."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from gaussian_next_mean_risk_metric_v1 import risk
from third_endpoint_span_v1 import input_basis, third_covariance_kernel
from third_endpoint_score_risk_v1 import ScoreRiskOperator

HERE = Path(__file__).resolve().parent
OUT = HERE / 'third_endpoint_score_risk_bound_v1.json'
assert not OUT.exists()
files = ['public_p2_third_cell_mini0057_v1.npz', 'public_p2_third_endpoint_span_v1.npz', 'third_endpoint_score_risk_wide_v1.npz']
loaded = []
provenance = {}
for name in files:
    path = HERE / name
    receipt = json.loads(path.with_suffix('.json').read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == receipt['output_sha256']
    provenance[name] = digest
    loaded.append((dict(np.load(path)), receipt))
(f, _), (old, _), (fit, fr) = loaded
with threadpool_limits(limits=3):
    mean = f['host'] + f['post13_head_delta'].astype(np.float64)
    u = mean / np.linalg.norm(mean)
    beta, v0 = fr['beta'], fr['frozen_variance']
    n = len(mean); tau2 = 2 / n
    factor = tau2 * tau2 * np.sqrt(1 - beta) / (8 * np.pi * v0)
    s = f['sigma']; R = f['Cpre'] / np.outer(s, s)
    H, powers, _ = input_basis(f['state'], R)
    kernel = third_covariance_kernel(R, s, f['alpha'], f['pdf'], f['gate'], f['gaussian_mean'] / s)
    op = ScoreRiskOperator(H, powers, kernel, u, beta, ridge=.001)
    rhs = op.adjoint(op.metric(old['target_covariance'] / op.covariance_scale)) / op.scale
    residual = op.normal(fit['theta']) - rhs
    rnorm = float(np.linalg.norm(residual))
    relative = rnorm / float(np.linalg.norm(rhs))
    assert np.isclose(relative, fr['solve']['relative_normal_residual'], rtol=1e-7, atol=0)
    target_risk = risk(old['target_covariance'], u, beta, factor)
    observed_risk = risk(old['target_covariance'] - fit['predicted_covariance'], u, beta, factor)
    assert np.isclose(observed_risk / target_risk, fr['analytic_risk_residual_ratio'], rtol=1e-12, atol=0)
    delta_transformed = rnorm / (2 * np.sqrt(.001))
    delta_risk_norm = 1.001 * np.sqrt(2 * factor) * op.covariance_scale * n * delta_transformed
    lower = max(np.sqrt(observed_risk) - delta_risk_norm, 0) ** 2 / target_risk
    result = dict(passed=True, exact_same_ridge_optimum_cannot_pass=bool(lower > .50),
                  normal_residual_norm=rnorm, relative_normal_residual=relative,
                  transformed_prediction_displacement_bound=delta_transformed,
                  inflated_physical_risk_norm_displacement_bound=delta_risk_norm,
                  observed_risk_residual_ratio=observed_risk / target_risk,
                  exact_same_ridge_optimum_risk_ratio_lower_bound=lower, required_risk_ratio=.50,
                  coefficient_updates=0, provenance=provenance, utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification='Deterministic algebraic same-ridge bound with1.001 numerical margin, not interval arithmetic, statistical confidence, an unregularized capacity bound or a claim about other predictors.')
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
