# Frozen-variance Gaussian risk audit: passed

Finished September 6, 2026, 05:17:14 PDT (receipt UTC 12:17:14).
Mini57 gregory-wilson, post13, the same fixed third-plane endpoint fit;
no new coefficient fit, target download, state write, or estimator score.

All six actual pre14 boundary arrays reproduce byte-exactly. Raw affine
mean was not stored; alpha, sigma, and Gaussian mean reproduce exactly,
and the alpha*sigma round-trip differs by at most 4.44e-16. Small explicit
Gaussian-fourth-moment certification has relative error <=3.58e-16.

Two preselected independent 4096-direction Gaussian WEIGHT streams give:

| Field | Analytic risk | Varying-variance Monte Carlo / analytic |
|---|---:|---:|
| Target covariance response | 4.3616661e-10 | 1.04629 |
| Fixed predicted response | 9.8543206e-11 | 1.03800 |
| Residual | 3.6153127e-10 | 1.05485 |

Frozen-variance analytic/MC differences are <=0.752 batch standard errors.
Analytic residual ratio .828883 versus varying-variance MC .835662
(batch SE .020874): relative disagreement .008112, passing the fixed .15
gate. Actual W14 ratio .859562, compared with ordinary covariance
Frobenius ratio .468996. Mean-ray suppression beta=.952658. This confirms
a substantial loss mismatch for this fixed field and training cell, not
that changing the loss necessarily creates a learnable correction.

All predeclared gates passed. The failed old K21-weighted ridge objective
and its same-objective optimum bound remain closed. This distinct,
analytically derived risk justifies one separate same-feature oracle fit
under a new preregistered decision. It does not justify a feature/regularizer
sweep, held-out target access, or submission.

Receipt: gaussian_next_mean_risk_metric_wide_v1.json. Output NPZ SHA256
ba906973f2d2a6757290d8f3a5ab539d6e36653047e1315c2dcd188af1d9c67a.
