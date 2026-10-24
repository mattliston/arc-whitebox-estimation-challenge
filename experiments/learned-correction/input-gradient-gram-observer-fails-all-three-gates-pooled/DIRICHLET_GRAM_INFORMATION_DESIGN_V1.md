# Bank-conditioned input-gradient Gram: fixed information pilot design

Phase 2 only, September 6, 2026 PDT. DESIGN, not execution authorization.
No network or target arrays have been opened for this proposal. Root must
accept independent source/cost review and a small numerical certificate
before generating rows. No existing/frozen estimator or data is changed.

## New information question and prior-work distinction

For a true prefix h(X), define its input Dirichlet Gram

    K_ij = E[grad_X h_i(X) dot grad_X h_j(X)].

For Gaussian input, orthogonal Hermite sectors give

    Cov(h_i,h_j) = sum_(k>=1) C^(k)_ij,
    K_ij = sum_(k>=1) k C^(k)_ij.

Thus K contains a degree-weighted angular/input-variation statistic, not a
second estimate of the mean. Mean/covariance alone do not determine it for
arbitrary functions. The planned K is an APPROXIMATION described below,
not an exact Dirichlet Gram of the real network. The test asks whether its
additional depth-integrated information predicts actual held-out final
mean error after conditioning on the same existing head inputs.

The independent Gaussian-shadow pilot propagated a separate mean/covariance
trajectory and failed. Its code `gaussian_shadow_disagreement_v1.py` contains
no input-gradient state. The signed matrix-error observer propagated fitted
error sources, not K. The twin-Gaussian symmetry proof concerned the marginal
mean/covariance obtained by re-closing the twin law; it did not test a learned
readout of its correlation derivative. The scalar Ward audit proves that
substituting unconditional gradient energy into a consistent Gaussian mean
formula is a no-op. This proposal does NOT do that substitution: it observes
the full K history and changes only a final learned readout if qualified.

Searches of the Astra/G22 source names and notes, tracker and mathematical
lanes found no previous input-Dirichlet-Gram observer. This is a bounded
source audit, not a claim of universal novelty. No Gaussian-shadow feature,
order, seed or fitting variant is reopened by this design.

## Fixed causal recurrence and its limitations

Use the current bank's already-captured on-policy pre-covariances, x and
sigma tapes in `rows_current_transport_shadow_v2/row_RRRR`. They are
rounded NumPy-instrument fields, not a claim of native D4 bit identity.
For layer l, reconstruct mu_l = float64(x_l[:,3])*float64(sigma_l).
Symmetrize the saved pre-covariance as (C+C.T)/2; its diagonal must be
strictly positive. Set s=sqrt(diag(C)), alpha=mu/s and rho=C/(s*s.T).
Set diag(rho)=1 explicitly; require every offdiagonal |rho|<1. No clipping,
PSD projection, jitter, parameter adaptation or discarded units.

At the first ReLU, mu_0 is identically zero. Initialize using the exact
Gaussian joint-gate formula under the supplied first Gram:

    Q0_ij = 1/4 + arcsin(rho0_ij)/(2*pi),
    K0 = Cpre0 * Q0.

For l=1,...,14 compute

    Kpre_l = float64(W32_l.T @ float32(K_(l-1)) @ W32_l),
    Kpre_l = (Kpre_l + Kpre_l.T)/2,
    K_l = Q_l * Kpre_l.

All matrices are full width. Q is the fixed Gaussian gate Hermite
approximation of degrees 0 through 5:

    b0 = Phi(alpha),
    br = (-1)^(r-1) He_(r-1)(alpha) phi(alpha), r=1,...,5,
    Q_ij = sum_(r=0,...,5) br_i br_j rho_ij^r / r!, i!=j,
    Q_ii = Phi(alpha_i).

These are exactly the derivatives of the existing Gaussian6 covariance
polynomial on offdiagonal entries with fixed means and variances. The exact
gate diagonal is retained separately. Q is not an exact offdiagonal
bivariate probability and no Fréchet-bound claim is made for its truncation.
On exact PSD correlation inputs, its Gram/Hadamard series plus the positive
unrepresented diagonal variance gives PSD. That conditional mathematical
fact is not a certified PSD statement about every rounded bank tape.

At l=15 compute full Kpre15 by the same two products and symmetrization,
but no outgoing gate kernel/K15. Require all intermediates finite and
every propagated diagonal strictly positive. No wide Cholesky or PSD claim
is part of this information pilot; no finite or positivity failure may be
repaired by changing the state law.

The factorization used after layer0 is the approximation

    E[gate_i gate_j * (Jpre_i dot Jpre_j)]
      approximately Q_ij * E[Jpre_i dot Jpre_j].

It drops correlations between gates and input-gradient products. Gaussian
gates are additionally evaluated under a bank-conditioned marginal law.
It is therefore NOT an exact input Jacobian calculation, not a derivative
of the complete P/Q cumulant card, and not a verified true two-input kernel.
The first layer's exactness does not prove fidelity at depth.

## Exactly eight final features; no current-target feedback

Use actual scalar-final bank fields from
`rows_matrix_error_observer_v1/row_RRRR/final.npz`, whose prefix/scalar-final
qualification is already recorded. Let sigma, alpha=x[:,3], gate=x[:,5]
and host denote their unstandardized stored scalar-final values. Let
C15 denote the symmetrized final FULL pre-covariance tape; only its
offdiagonal is consumed. Its diagonal arithmetic is not the scalar-final
variance computation, and must not replace that variance.

    rho_ij = C15_ij/(sigma_i sigma_j), diagonal set to zero,
    D_ij = (Kpre15_ij-C15_ij)/(sigma_i sigma_j), diagonal zero.

Features in this order:

1. log(diag(Kpre15)/sigma^2);
2. mean_j D_ij;
3. sqrt(mean_j D_ij^2);
4. (D @ alpha)_i/n;
5. (D @ (host/sigma))_i/n;
6. (D @ gate)_i/n;
7. mean_j D_ij*rho_ij;
8. mean_j D_ij*rho_ij^2.

Everything is computed before changing the final head. K never changes
bank means, covariance, CP factors, masks, source features or recurrent
state. Each new row copies existing head input h, host, sigma and original
predictions byte-for-byte, adds only `sources`, and seals source/input
hashes. There are no per-layer fitted coefficients or teacher state writes.

## Cost and availability are part of the question

Thirty extra full FP32 products cost 60*n^3 = 64,424,509,440 classical
multiply/add FLOPs at n1024; the bank's first Gram is shared. Add first
arcsin, 14 six-term gate maps, 15 symmetrizations, covariance normalization,
casts/copies, all scalar/reduction work and the eight features.

IMPORTANT: the current scalar-final bank does NOT expose full Cpre15.
The archived full tape is not a free inference datum. An implementation
reusing the live first product in `_cov_compact_diagonal` still requires
an additional active congruence product and cold diagonal Gram to produce
the offdiagonal final covariance: about 2*n^3 more FP32 work plus gathers,
scaling and assembly. Without that internal reuse it costs more. Preserve
the original FP64 scalar variance independently. The information test uses
the archived offdiagonal, so a pass must still establish native acquisition
and numerical feature parity, not simply attach this tape to a candidate.

The leading 62*n^3 subtotal is 66.572B, utilization 0.0302734375, before
all other work. This is not a native flopscope bill or wall-time result.
Buffers for K and gate powers are O(n^2), plus optional C15 materialization;
no depth-long derivative tape is required in inference. The pilot must show
a large raw gain to justify the extra trajectory and failure risk.

## Fixed prerequisite and one fit

Small numerical certificate, no real network/target arrays: seed930079,
synthetic widths9,17,31, three SPD fixtures each. Compare offdiagonal Q to
the central finite difference of Gaussian6 covariance at fixed marginal
means/variances, relative/max tolerance 1e-7 with perturbation 1e-5 times
the corresponding sigma product. Check symmetry, exact gate diagonal,
positive semidefiniteness on these SPD fixtures to -1e-10, first-layer
arcsin values at correlations 0, +/-1/2, +/-sqrt(1/2), and a single-layer
linear algebra transport/positive hidden-unit scaling identity to 1e-10
in float64. These checks certify the stated implementation, not accuracy
against a true deep Jacobian or the board.

Only after root acceptance of certificate and independent review: generate
input-only smoke rows35/100 and check deterministic replay plus unchanged
parent bytes. Then generate the fixed24 ranks35..58, at most two nice-10
CPU workers with two BLAS threads each. Every row must pass source/input
hash, domain, finite, positive-diagonal and unchanged-bank qualification
before ANY new training target read. No samples or new truth acquisition.

The single matched fit uses TRAIN20 and VALID37/39/42/48, which are reused
research holdouts, not the independent14-net release panel. Both arms are
134->128->128->1 final heads initialized from the current Q's126 columns;
the eight added weight columns start at zero. Control gets zero extra inputs.
TRAIN-only feature center/std, standard-deviation floor1e-6, clip to +/-8.
Fixed seed930080, 40 epochs AdamW, lr5e-4 cosine to5e-5, weight decay2e-5,
batch8192, gradient norm cap5, final epoch only. No model selection. Train
only on existing1e9 final truths for TRAIN. Seal BOTH fitted payloads and
all held-out predictions before reading VALID truth arrays or losses.
Evaluate with ordinary NumPy final_delta and the actual output transform;
NumPy/MPS deltas must agree within2e-9. One MPS job, fallback disabled,
two CPU threads, no threadpoolctl import in AP, fit cap600s/RSS6GiB.

Advance only if pooled raw source/bank<=0.65, source/control<=0.70 and at
least3/4 bank wins. Failure closes this fixed observer/features/order/window/
fit: no added channels, altered moment kernel, seeds, scales, extra epochs,
capacity, state writes or expanded fit. A pass still needs200-net training,
native feature/candidate parity and full metering, both independent release
panels, persistent official-client wall/residual/memory validation and a
hashed frozen submission. No score crossing may be claimed from this pilot.
