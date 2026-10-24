# Root acceptance and qualifications — September 6, 2026

Phase 2 only. Root accepts the frozen one-rank protocol in
REFLECTED_RESIDUAL_COPULA_TREE_PROTOCOL_V1.md for a separately reviewed
observation-only rank35 recapture and one numerical/fidelity judge. No
additional ranks, alternate root/order, fit, repair, rollout, candidate or
submission is authorized by this acceptance. All sources are frozen before
their first real-state execution. Existing receipts and experiments remain
unchanged. The active score objective is not achieved or externally blocked.

The two independent source reviews found the scalar derivative and strict
monotonicity proof, stable marginal formulas, reflected Hermite coefficients,
Price integral, tree action and mask-before-trace conventions correct.
The synthetic receipts certify only their finite fixtures. In particular,
the scalar inverse fixture is a same-function round trip; sampled derivative
signs are not independent finite-difference verification of the identity.

The degree12 pair polynomial is NOT generally monotone. A source-only
counterexample is identical alpha=-3 near latent correlation -1, where the
polynomial derivative is negative. Sign-bracketed bisection still locates
a polynomial root by continuity, but neither uniqueness nor feasibility of
the exact model follows. The judge separately checks the polynomial bracket,
exact one-dimensional endpoint covariance, fixed32/64 Price integration,
and latent positive definiteness, with no repair or alternative spelling.
A failed polynomial bracket closes this numerical arm, not all copula
models. Mixed Gaussian pairs use C/(sigma_G*a_other), without an extra sign.

For completeness, at ordinary latent endpoints, with m_i the ordinary
shifted-ReLU mean, the independent product formulas are:

    E[Y_i Y_j] at +1 = (1+alpha_i*alpha_j)*Phi(min(alpha_i,alpha_j))
                       +max(alpha_i,alpha_j)*phi(min(alpha_i,alpha_j))
    E[Y_i Y_j] at -1 = (alpha_i*alpha_j-1)*(Phi(alpha_j)-Phi(-alpha_i))
                       +alpha_j*phi(alpha_i)+alpha_i*phi(alpha_j)

The latter is zero when alpha_i+alpha_j<=0. Subtract m_i*m_j and apply
the explicit reflection/scales. These follow by one-dimensional integration
over their intersection support. The 1e-6 standardized endpoint allowance
is the protocol's covariance-residual tolerance, not clipping or root repair.

The arithmetic subtotal in the agent draft includes only ONE Cholesky per
transition, whereas the present diagnostic checks BOTH C_aux and latent R.
If both guards remain deployed, the second costs another 10,737,418,240
FP64 metered-like FLOPs over15 full-width layers, or 0.0048828125 utilization.
Thus the corresponding full-product/two-factorization subtotal is
281,320,357,888 FLOPs, utilization0.1279296875, BEFORE the other explicitly
excluded maps/inversions/heads/copies/guard costs. This is unmetered and
contingent on usable FP32 tree/star products. It is not a score or runtime
claim. No such claim is allowed from the FP64 one-action teacher test.

Root's fixed judge has a separate causal predictor file and a timestamped
hash seal before decoding the sole pre13 K21 loss array. Acquisition's
teacher reads only verify unchanged-bank parity; they do not influence
the completion. The learned response-row branch's requested one-round
exception is still awaiting user approval and is NOT authorized here.
