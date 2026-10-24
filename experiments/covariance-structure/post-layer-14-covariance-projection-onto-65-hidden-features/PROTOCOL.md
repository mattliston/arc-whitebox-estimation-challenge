# Terminal covariance supervision: fixed prerequisite

Phase 2, 2026-09-20. Written before this study decodes arrays. The accepted
Lazy source and archive remain unchanged. Official #331599 is the bank;
qualified future submissions are preauthorized. This study cannot qualify a
submission or establish a deployable covariance supplier.

The preceding covariance transport screen found substantial descriptive
one-step value but insufficient free propagation from layer 7. Here the
question is narrower: can the current terminal history subspace represent
the *measured cumulative covariance error* in a way useful to the complete
final scalar/GNN calculation? Covariance labels choose the projection;
final mean labels never choose coefficients.

This is a related bilinear history family, not a new representation family.
The old endpoint-history decoder used P0 with oracle previous moments and
learned one-step covariance innovations across layers. It failed its next
mean gate. The current H14 bilinear readout fitted final means directly and
also failed. Neither result is reopened or relabeled. Covariance supervision
of current autonomous cumulative error is a distinct target/interface, but
has no assumed advantage. First test its privileged physical projection;
do not launch another learned fit merely because labels are available.

Reuse the sealed TCI 35/100/35 terminal captures, source/guard receipts and
existing coherent 8M covariance references. Rank 35 also has its independent
8M stream. Use ordinary original 1B final means only after all new predictions
are sealed. These are reused development networks, not fresh validation.
Authenticate every input and source before numerical work. Require P=8M and
post14 to pre15 mean/covariance affine consistency <=1e-4 relative.

Let I be the original post14 active set and C the original masked covariance.
Fix X_I = sqrt(C_ii) * [H14_i,1], and X=0 on cold rows. Require positive
active diagonals; no floors or repairs. Normalize its columns by their RMS
(floor 1e-12) and take the left singular vectors with singular value greater
than 1e-7 times the largest. This is target-free geometry, not a rank search.
For each covariance reference define E_I,I = sym(Ctrue_I,I - C_I,I).
The projected correction is Q (Q.T E Q) Q.T, active rows only, with no cold
diagonal correction. This is the exact Frobenius projection into that space,
not the best final-error correction or a fitted shared decoder.

Compare exactly two fixed privileged interventions:

1. Projected: add the above active covariance correction to the original C.
2. Full reference control: substitute true covariance on the original active
   block and true variances on cold diagonals; keep other cold entries zero.

Earlier means, final mu, CP/star terms, bpre, radial, H14 and all learned
parameters remain fixed. Recompute final v/active_v, trace k3, radial k4,
nonlinear scalar moments, all final features, full GNN and nonnegative clamp
with the original native FNP helpers. This is a finite local replay; it is not
a coherent full moment law or a complete estimator. Require all final pre
variances >=1e-10, finite values, and native zero replay maxabs<=1e-6 and
RMS<=1e-7 versus the recorded parent. No strength/PSD/precision rescue.
Repeat rank35 projected stream A once and require exact outputs and bills.

First certify the projector on deterministic small full/subrank/signed
fixtures: symmetry, idempotence, residual orthogonality and Pythagorean
identity <=1e-10 relative. Native replay setup and all calls are metered,
but truth/projection construction is research work: no deployable bill or
adjusted score is inferred. At most nine native final-block replays, no full
estimator calls, fitting, new samples or new dense trajectory captures.

Seal all predictions and projection diagnostics before opening 1B means.
Report covariance Frobenius and final-variance response errors, and final
MSE ratios against those references. For rank35 also report cross products
of residuals formed separately with each stream's final mean and prediction,
including observed = cross + half-difference noise identities. Independence
removes leading sampling variance from linear cross products, but shared
MPS bias and finite nonlinear replay bias remain; these are point estimates,
not confidence bounds or unbiased population guarantees.

Advance only if the PROJECTED intervention has ordinary 1B MSE/parent <=.80
for each of 35-A,35-B,100-A, and positive-denominator rank35 cross-residual
ratio <=.75. The full-reference control is descriptive and cannot pass on
the projected arm's behalf. Failure closes this fixed prerequisite: no
rank, cutoff, feature, scale, cold support, fit or reference expansion.
A pass merely licenses designing a distinct shared covariance-supervised
supplier with proper held-network and native cost checks.

One owned numerical group, nice10, six BLAS thread controls1, 8GB group RSS,
zero added swap. Existing guard has a 256MiB emergency swap stop. Unit30s,
screen120s, <=20MB new artifacts. Preserve failures. No external messages
or upload. The approximately 1e-9 goal remains active.
