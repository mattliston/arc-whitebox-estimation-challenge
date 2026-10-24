# Conditional quadratic tensor information gate

September5,2026. Phase2 only. No new bank or milestone; objective<1e-9.

## Distinct observable, not a repeat of closed feature sets

The earlier conditional-skew-energy probe used T_ii,alpha, obtainable from
the K21 plane. The earlier compact tensor-direction head used T_i,u,u along
two mean/covariance-generated directions on the older compact host. Neither
computed the quadratic conditional-mean energy/cumulants from the matrix
T_i,alpha,beta across dominant covariance modes on the current source/full
pair carrier. Their failed fits remain closed; no sweeps of them are run.

Let Q contain8 top covariance eigenvectors divided by sqrt(eigenvalue).
For each unit i, form B_i[alpha,beta]=kappa3(Z_i,Q_alpha.Z,Q_beta.Z)/sigma_i
from the actual carried CP+star+trace tensor. Let c_i=Cov(Z_i,Q.Z)/sigma_i.
The Gaussian conditional quadratic approximation

    z_i/sigma_i ~ c_i.x + .5*(x^T B_i x - trace(B_i))

has quadratic variance .5*tr(B_i^2), third cumulant
3*c_i^T B_i c_i + tr(B_i^3), and fourth cumulant
12*c_i^T B_i^2 c_i + 3*tr(B_i^4). These are new JOINT observables;
they are not added to the physical variance (which would double count).
Treat them only as information for a fixed residual probe.

Use8 rotation/sign-invariant features: trB, tr(B^2), (trB)^2,
tr(B^3), tr(B^4), c^T B c, c^T B^2 c, c^T c. All are frozen before
the current target correction. Exact partial eigensolver is an offline
information ceiling; no cheap/billed eigendecomposition is presumed.

## Fixed experiment

First certify CP+star+trace contractions against a dense synthetic tensor.
Then24 existing honest teacher rows35..58,20train, held-out37/39/42/48,
layers10..15 only. Byte-verify old features, host and head outputs against
existing rows. Do not change the carrier trajectory or read sealed targets.

Same-capacity134->64->64->1 residual probe, eight real features versus
eight zero columns. Same initialization, zero final layer,40epochs final
epoch only, AdamW1e-3->1e-4 cosine, weight decay2e-5, gradient cap5,
seed951059. TRAIN-only feature RMS and loss normalization. Require joint
held-out MSE <=.92 of zero-feature control and <=.95 of frozen parent,
with at least3/4 nets improved. If it fails, no mode/rank/basis/feature
sweep, extra labels, larger fit, port or submission in this probe family.

Only passing information could justify a separately costed co-adapted
carrier. A teacher residual result is not final-layer adjusted accuracy.
At most3 nice10 CPU row workers with2BLAS threads; one MPS job, stable swap.
