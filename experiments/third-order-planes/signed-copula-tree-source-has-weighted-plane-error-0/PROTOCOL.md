# Fixed signed copula-tree action v1

Phase2 only, September8,2026. This is one separately versioned, no-fit,
old-rank35/post12 source-fidelity test under the user's broad experiment
authority. It preserves the original degree12 inverse failure and the exact
endpoint verdict. It is NOT their hidden retry, an exact covariance-matching
model, a positive joint law, a current409f candidate, or a raw-score test.
Source review and root acceptance must precede the sole --execute invocation.

## Fixed information and change of hypothesis

Use only the already sealed reflected-copula causal capture, its unchanged
marginal parameters, and the exact-endpoint classification. The latter found
345678 feasible pairs and18 strictly below their lower endpoint, with no
upper violations or unresolved pairs. All18 are same-reflection pairs.
Set their formal R entries to -1. Do not move the physical covariance.
No Cholesky, eigenvalue screen, PSD projection, jitter or interior relaxation
is required or performed: the tensor formulas are defined for any finite
symmetric R with unit diagonal. Clipping is the canonical elementwise
least-squares extension of the monotone covariance map, not exact matching.

The other345678 entries use exactly26 sign-bisection iterations on [-1,1]
with64-node Gauss-Legendre Price covariance. There is no polynomial inverse,
root selection, additional iteration, order/seed/support sweep or fallback.
This numerically approximates the exact covariance inverse; agreement of
32/64 quadrature is a numerical check, NOT a rigorous quadrature enclosure.
The18 certified endpoint classifications remain rigorous independently.

The frozen alpha/eta/tau/sigma/a/q arrays are not refit. The Price formula,
for ordinary correlation r and standard-normal CDF Phi, is

    F(r) = r Phi(alpha_i) Phi(alpha_j)
         + r^2/(2 pi) integral_0^1 (1-u)
           exp(-(alpha_i^2-2 r u alpha_i alpha_j+alpha_j^2)
               /(2(1-r^2 u^2))) / sqrt(1-r^2 u^2) du.

The standardized reflected covariance is
eta_i eta_j (tau_i/sigma_i)(tau_j/sigma_j) F(eta_i eta_j R_ij).
CDF values use scalar libm erfc; all matrix/vector arithmetic is FP64.
The scientific target is the literal auxiliary covariance divided by the
stored-sigma product. Its FP64 evaluation must lie in the prior certified
target interval. The original covariance is symmetrized only for this
auxiliary inference, exactly as in the frozen capture protocol.

Final32/64 covariance discrepancies must be <=1e-8 standardized on EVERY
pair. The64-node target residual must be <=1e-6 on UNCLIPPED pairs only.
For clipped pairs report their nonzero target residual, and require the
64-node value to lie within1e-8 of the certified endpoint interval. Never
force a clipped pair to claim a zero or <=1e-6 target residual.
Record all extrema and partial progress before enforcing a failed guard.
Any numerical failure stops this spelling; no automatic retry or tuning.

## Formal tensor and exact current-owner restoration

Let S be the captured actual total post K21. Retain captured B and b:

    trace21_ij = B_ii b_j + 2 B_ij b_i,
    E21 = S - trace21,
    U = diag(a) R on the original live block, zero elsewhere,
    Ttree = sum_i q_i SymSum(e_i, U_:i, U_:i),
    tree21_ij = U_ij^2 q_j + 2 q_i a_i U_ji,
    A21 = E21 - tree21,
    Tnew = Ttree + Star(A21) + original trace.

The tree is the known two-edge Gaussian/Hermite birth sector for the
declared first and second response coefficients, not the full exact
Gaussian ReLU third tensor. In particular it does not include the
three-residual/triangle sector or establish a complete nonlinear law.

The repeated plane is restored to the original total plane in real
arithmetic. Require relative restoration <=1e-10; report normalized maximum
entry error. The original star is already included in S and is not added
again. No physical C, scalar mean, radial coefficient, trace, mask or head
is changed. The actual small covariance asymmetry is retained in its
captured trace reads; no claim of an exactly symmetric probability tensor
is made for those rounded trace conventions.

For the actual W13 and rows J=(0,32,...,992), compute the full X=W13.T U
and Q=(A21-(2/3)diag(diag(A21))) W13. Use the original two tree and two
star rectangular products for T(w_j,w_j,w_k). Mask the nontrace block by
the captured pre13 alive mask, preserve its complete diagonal, THEN add
the actual captured pre13_Bpre/pre13_bpre trace. Do not recompute that
trace from a symmetrized covariance. Freeze the entire predictor, R/U/q,
restoring plane and consumer weights in an exclusive NPZ and hash seal
BEFORE decoding the original pre13_K21 loss array.

The only information gate is the old fixed Gaussian-pair-weighted fidelity

    weight_jk = .5 phi(alpha13_j)/sigma13_j * Phi(alpha13_k),
    fidelity = norm(weight*(prediction-teacher))/norm(weight*teacher)
               <= .03.

The finite nonzero denominator is mandatory. Report unweighted and diagonal
errors but do not use them to choose a different model. This strict gate is
an interesting replacement diagnostic, not a necessary condition for lower
raw error. The old0.337814/0.316703 covariance-metric errors are contextual:
that older replay used a different covariance/trace arithmetic frame and
fixed physical-correlation squared legs. It is not a paired raw comparator.
The new skew-conditioned inverse changes both squared legs, rather than
rerunning that old PCG coefficient fit. Useful fidelity is unmeasured.

## Qualification, scope and resources

Before decoding any causal array, run one fixed n4 dense-tensor fixture
with an explicitly indefinite R, signed q, nonsymmetric repeated plane,
dyadic W and a nontrivial next mask. Compare current-plane restoration and
the masked tree/star-plus-trace read with literal dense cubic contraction
at1e-12 times max(1,reference maximum). Also compare32/64 Price evaluations
for five fixed alpha=0 correlations to the analytic arc-cosine formula at
1e-12. These finite fixtures do not establish a general inverse error bound.

One fresh OFFPY process, nice>=10, explicit BLAS1 launch controls, no SciPy,
Torch, MPS, fit, network simulation or concurrent numeric worker. Fixed
120s wall watchdog, peak RSS<=1000000000 bytes, pair chunks<=16384. No latent
eigendecomposition is performed. Check resources after each pair chunk and
major stage. All outputs are exclusive under result_v1; preserve failure.
Hash sources and input bytes before and after. The teacher archive may be
hashed before the predictor seal but only its designated loss field is
decoded after sealing. No MC truth or final mean reference is accessed.

This diagnostic reuses already-paid archived acquisition; it does not hide
that acquisition in a future deployment bill. Its full-width action uses
two1024-square and four32-by1024 rectangular FP64 products, plus inversion,
all elementwise work, checks and IO. The product-only classical arithmetic
is 2*n*n*(2*n-1)+4*32*n*(2*n-1); ordinary FP64 metering would multiply that
subtotal by2. It is not a measured or complete FNP bill. Record actual
Price pair/node counts separately. A full all-layer tree/star model would
still own all marginal/inverse work, six third-action products plus two
covariance products per transition, radial/scalar updates, ALL current P
heads, masks/copies/initialization and native numerical qualification.

Even a PASS only supports a separate current-frame/full-cost investigation.
It is neither current-bank compatibility, endogenous stability, final raw
accuracy, a panel pass, a milestone nor permission to submit. No broader
copula or signed-tensor family is closed by this one fixed outcome.
