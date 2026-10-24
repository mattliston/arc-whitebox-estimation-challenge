# Nonlinear historical pair memory: one fixed information test

Phase 2 only, September 6, 2026. Root's selected test, fixed before any new
bank-action capture or target/feature evaluation. The two-agent architecture
proposals are orientation; this document alone fixes the first experiment.
No MC truth, final raw fitting, full student, native port or submission.

## Scope, ownership and mathematical question

Ranks 35–58. Train on the twenty ranks excluding held-out 37,39,42,48.
Only the query post12 -> pre13 is scored. History uses post0..12. These are
public-development networks, not a claim of pristine competition holdouts.

Use a new coherent replay of the unchanged NumPy full-pair bank with original
P0 and actual separate final Q. Capture actual represented physical post K21
(float32 CP/star state interpreted in float64 plus the actual trace tensor),
not just the nonlinear routine's target before head/gating/rounding. Preserve
the finite-precision difference as a diagnostic. Pre13 K21 is the bank's
actual read and is captured before nonlinear/head changes. Frozen bank output
parity must pass before these data are used. Current targets are loss-only.

At each prior boundary, the actual bank post covariance diagonal defines the
positive units s_post=sqrt(max(diag(Cpost),1e-10)); D=K21/(s_i^2*s_j).
This explicit floor defines units, not a covariance/PSD repair. Capture and
report the actual minimum diagonal/negative counts. Pre units are the bank's
actual sigma. m,C,D history is teacher-forced, an optimistic information grant.
The learner will not be advertised as a self-consistent CP-free rollout.

## Fixed source, transport, and exactly matched control

N(A)=A/max(RMS(A),1e-3), with RMS over all n*n entries. offdiag sets only
the diagonal to zero. These normalizations use predictor inputs, never targets.

For post-layer D_l and d_l=diag(D_l), define

    B_l = offdiag((diag(d_l) D_l + D_l.T diag(d_l))/2).

History arm starts H_-1=0. Set U_0=0. For l=1..12,

    V_l = diag(s_post,l-1) W_l diag(1/s_pre,l)
    U_l = V_l.T H_l-1 V_l
    H_l = tanh(.5 N(U_l) + .5 N(B_l)).

At l=0 use H_0=tanh(.5 N(B_0)). The RESET arm is independently constructed
as H_l=tanh(.5 N(B_l)); thus it keeps the same immediate pair source but no
older memory. Both arms receive identical bank inputs. Neither has a learned
memory update, source coefficient, decay, selected layer, or learned basis.

At query13, V=diag(s_post,12) W13 diag(1/s_pre,13). Compute the exact
coordinate-star action once:

    E=D12-(2/3)diag(diag D12); Q=E V
    K=(V*V).T Q + 2(V*Q).T V.

For each arm U=V.T H12 V, J=K U. K U has correct affine repeated-third parity;
U K does not. This is a typed affine-action fact, not a claim that H represents
a full third tensor or that a complete learned network has sign symmetry.

Let M be the actual bank pre13 live-pair mask with its diagonal zero. Fixed
features are F0=N(M*K), F1=N(M*J). The shared K and full U/J are computed before
this output-domain mask; no intermediate summation direction is selected.
Target R=M*(Dpre13_bank-K). The two arms fit only two shared coefficients each:
Rhat=beta0 F0+beta1 F1. Current pre13 target values never enter features, H,
mask, scales or normalization. The diagonal is not fitted or supplied to H.

## Exact frozen covariance consumer, not merely its rho-zero term

The initial agent sketch omitted the bank's r=1,2 cubic correlation terms.
Before data access root replaces it with the FULL frozen off-diagonal K21
response of radial_mehler.make_nonlinear(6,2,True,True). A tiny independent
finite-difference certificate is required. Other moments, learned mean
feedback and current marginals remain fixed.

For deltaD, set deltaK_ij=s_i^2*s_j*deltaD_ij. Use the same bank pre covariance,
original variance denominator, sigma, rho, alpha, pdf and gate. With
A1=g, A2=pdf, A3=-alpha*pdf, A4=(alpha^2-1)*pdf,

    T_ij = sum(r=0..2) rho_ij^r/r! * A[r+2]_i*A[r+1]_j
    L(deltaD) = .5 outer(s,s) *
               [T*(s^2/diag(Cpre))[:,None]*deltaD + transpose].

The final notation means transpose of the entire first bracketed matrix;
the sigma outer product is symmetric. Set its diagonal to zero. Retaining
s^2/diag(Cpre) matches the bank if its sigma floor was active. Invalid or zero
original variance is a qualification failure, not a silently repaired input.

The next consumer is diag(W14.T L(deltaD) W14) times the bank pre14
pdf/(2*sigma). It is a leading Gaussian variance-to-mean response, not the full
bank/head Jacobian, new Monte Carlo truth, or final raw score. Omitting diagonal
third error and endogenous state drift makes a pass necessary evidence only,
not sufficient accuracy evidence for a future replacement estimator.

## One two-variable fit and frozen decision

For each arm compute covariance and next-mean response features/targets.
Normalize the two squared-error losses by their TRAIN-only zero-correction
energies, each averaged equally over the twenty nets and coordinates. Sum
the two normalized losses. Solve the two-by-two normal equations once, with
ridge 1e-6*trace(G)/2 added to the diagonal. No epochs or optimizer search.
Coefficients are common across neurons/networks; held-outs never select them.

Report each held-out net and pooled held-out ratios, absolute errors, feature
norms, coefficients and Gram conditioning. Required evidence is ALL:

- History pooled covariance and next-mean residual squared-error ratios <=.50
  versus zero correction.
- History pooled next-mean residual error <=.80 times RESET.
- History beats RESET in next-mean residual error on at least three of four nets.

Zero/nonfinite normalization energies or an invalid solve are inconclusive.
No source, decay, normalization, layer, rank, features, ridge or coefficient
sweep follows a failure. A pass only earns a separate endogenous-state design
review, not an automatic fit or a submission. This is not a general impossibility
test for other nonlinear matrix memories.

## Costs and resource boundary

History source uses two full products per transition plus pointwise work;
the query uses the certified three-product star action and one K U product
per arm. Every teacher replay and offline consumer is recorded as offline
diagnostic work, not hidden estimator cost. Classical full replacement with
covariance2+star3+memory2+KU1+PSDGram1 needs nine products per transition.
With the provisional 20B map/head allowance it is about312.06B/.14191 budget;
that allowance is NOT metered or yet certified. It suggests sub4e-9 only if
bank-class raw survives; it does not establish sub3e-9 or the <1e-9 goal.

Acquisition: first rank35 only, one nice10 NumPy worker, two BLAS threads,
no Torch/MPS, preserve all artifacts, explicit root-reviewed execute flag.
Other fixed23 ranks only after the first coherent capture passes. Process
peak RSS must remain <=6GiB. No sources edited during acquisition or judgment.
Freeze source, weight and P/Q digests and recheck before artifacts are accepted.
Feature extraction/fit uses the same bounded NumPy policy; no large neural
training job is part of this round.
