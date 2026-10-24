# Algebraically centered moment arithmetic for differentiable fidelity

September6,2026, Phase2. The existing CPU full carrier uses f64 scalar and
moment arithmetic with f32 matrix pushes; it is locally derivative-certified.
The all-f32 MPS version failed the baseline accuracy gate. Its code still
forms r3-3*m*r2+2*m^3, r4-4*m*r3+6*m^2*r2-3*m^4-3*v^2, and pair differences
that cancel large mean-dependent terms. This is a NUMERICAL re-expression
of the same moment card, not a new closure, source, or learned model.

First certify scalar marginals against120-digit raw-moment arithmetic,
with fixed standardized means[-12,-8,-4,-2,-.5,0,.5,2,4,8,12], sigmas .3/1.7,
and fixed standardized(k3,k4) triples(0,0),(.02,.03),(-.015,-.02).
Use negative-tail Gaussian ReLU raw moments and exact Gaussian/reflection
cumulant identities for positive means to avoid large-polynomial
cancellation. Add Edgeworth corrections centered about Gaussian mean mg,
then recenter by the small delta=m-mg. Require all mean/variance/k3/k4
absolute errors normalized by sigma^order <=1e-11 versus high precision.
No parameter fitting or targets in this certificate.

Second, before any Torch port, rewrite pair cumulants using centered
Hermite derivatives F_r=B_r-2*(m/sigma)*A_r directly:
postK21=pair(F,A)*sigma_i^2*sigma_j,
postK22=pair(F,F)*sigma_i^2*sigma_j^2-2*cov^2.
Use the stable F1=2[(1-g)*mg/sigma-g*delta/sigma], with upper-tail probability
computed directly, and stable beta=2[(1-g)*mg-g*delta] for paired births.
Require the same f64 algebra on fixed synthetic small maps before wide
parent parity; preserve the original scripts and failed MPS receipts.

A future Torch/f32 port must preserve all original physics, active/birth
membership rules, P and Q payloads. It requires full bank baseline parity
on BOTH35/100 at the original2e-6 maximum mean tolerance, exact selection
membership auditing, finite-difference/local-autodiff checks and shared
machine resource checks before any training. The safer erfc form of the
normal CDF also needs backend support verified, not assumed. MPS supports
no f64 here; do not silently use a CPU fallback and call it a GPU fix.

Passing arithmetic fidelity would enable a separately justified training
experiment, not reopen all closed encoder/covariance fits. It establishes
no raw/adjusted gain and authorizes no estimator submission by itself.
AP Torch must never import threadpoolctl; no MPS work while another user
job is using that device. No new source download, MC, panel or targets here.
