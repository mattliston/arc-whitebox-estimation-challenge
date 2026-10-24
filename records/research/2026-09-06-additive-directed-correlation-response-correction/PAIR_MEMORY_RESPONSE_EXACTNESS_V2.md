# Additive directed-correlation response correction

Phase 2 only. September 6, 2026. Written before any pair-memory feature,
action-loss, coefficient, or held-out evaluation. No scientific arm changes.
The frozen v1 protocol, primitives, certificate, and captures remain intact.

The v1 response uses half + half.T. This is algebraically exact when the
input covariance is symmetric, as in its tiny certificate. The bank does
not explicitly symmetrize its float32 W.T C W result, however. In the
unchanged radial_mehler pair routine, BOTH directed K21 terms at (i,j)
are multiplied by rho_ij**r, not one by rho_ji**r. Therefore v1's symmetric
transpose is not the exact directed response for a nonsymmetric stored C.
This is a source-derived finite-precision ownership correction, not an
accuracy-target finding. No size of this effect on the wide inputs has
been measured by the author.

## Exact additive v2 formula

Keep the original C unchanged. With s, alpha, pdf, gate from the same bank
prestate, let v=diag(C), rho=offdiag(C/outer(s,s)), and

    A1=gate, A2=pdf, A3=-alpha*pdf, A4=(alpha**2-1)*pdf
    P_ij = deltaD_ij * s_i**2/v_i
    T_ij = A2_i*A1_j + rho_ij*A3_i*A2_j
           + .5*rho_ij**2*A4_i*A3_j
    S_ij = A1_i*A2_j + rho_ij*A2_i*A3_j
           + .5*rho_ij**2*A3_i*A4_j
    L_ij = .5*s_i*s_j*(T_ij*P_ij + S_ij*P_ji),  i != j
    L_ii = 0.

S swaps derivative endpoints at the SAME directed correlation. It is not
T.T unless the correlation is symmetric. P retains the original-variance
denominator even when the bank's sigma floor is active. Original variances
must be finite and strictly positive; no covariance, symmetry, PSD, or
response repair is introduced. The result need not be exactly symmetric.
It is an algebraically exact affine response of the frozen pair program;
different floating-point evaluation order is not a byte-identity claim.

pair_memory_action_physics_v2.py re-exports the UNCHANGED v1 normalization,
memory source, star action and next-mean consumer. Only the covariance
response changes. The history/reset recurrence, pre13 live domain, two
features, 20/4 split, ridge, loss normalization and .50/.80/3-of-4 gates
are unchanged. Both arms receive the same corrected operator. Prior
post-layer diagonals still enter B; the current query diagonal remains
excluded from predictions and the action loss.

No GEMM is added: the correction requires only a second pointwise
coefficient matrix and O(n**2) arithmetic/workspace. The complete-student
cost allowance remains provisional. Every wide result must report the
actual pre13 covariance antisymmetry, without repairing it or using it
to choose a model. The next consumer remains the declared leading
Gaussian variance-to-mean response, not the full bank/head Jacobian.

## Fixed tiny certification, not a value experiment

certify_pair_memory_response_v2.py preserves the six original symmetric
fixture constructions and adds two deterministic antisymmetric fixtures,
one active-sigma-floor symmetric fixture and one combined floor/asymmetry
fixture. The RNG seed is 930119. There is no target, wide input or fit.
It checks the new operator against direct differences of unchanged
radial_mehler.make_nonlinear(6,2,True,True), and checks half-step linearity.
The fixed relative tolerance is 2e-11. Symmetric fixtures must also agree
with v1 to that tolerance; at least one asymmetric fixture must exhibit
v1's nonzero discrepancy above 1e-10. No fixture, perturbation, seed or
tolerance sweep is authorized. The certificate source is provided for
root review and execution; its author has not run it.

The judge must pin both primitive versions, this note and the new tiny
certificate source; require the passing v2 certificate; and recheck all
digests before and after judgment. Separately, the resource-only capture
replay must pass two_memory_capture_replay_v2_qualification.json for exactly
35,36,37 before any v2 action capture is admitted to the fixed information
test. Neither technical correction constitutes a successful estimator,
new raw result, expanded fit, or submission authorization.
