# Fixed radial fourth-order birth direction

This tests one new physical field on I95, without changing the frozen learned
coefficients, head, gates, ranks, or compression schedule by hand.

The existing birth-base expansion contains the third-order Stein terms
`C_ij * skew_i * H2(alpha_i) * phi_i / 6` and `.5*d_i*K21_ij`.
For a centered input fluctuation, the first-order fourth-cumulant contribution
to `Cov(ReLU(mu_i+X_i), X_j)` has two terms:

    C_ij * excess_i * (3*alpha_i-alpha_i**3) * phi_i / 24
    + K31_ij * (-alpha_i*phi_i / v_i) / 6.

Under the card's existing radial fourth law, `K31_ij=excess_i*v_i*C_ij`.
Their sum is therefore

    F5_ij = -C_ij * excess_i * (alpha_i+alpha_i**3) * phi_i / 24.

An independent one-dimensional Gaussian integral of `z*ReLU(alpha+z)*H4(z)`
checks this sign and coefficient. This is an additional *birth-base* direction,
not a complete fourth-cumulant closure or a claim of an exact conditional law.
As with the original four directions, zero its diagonal and remove its column
projection on the original base B. Add it with fixed coefficient1. The original
stored-FP32 delta21 repair then preserves ownership of the changed newborn's
repeated plane. No separate retrofit of an oracle K21 plane is performed.

Prior archive search found Gaussian third-edge and direct plane corrections,
but no executed version of this fourth-order radial Stein birth field. This
limited absence check is not evidence that the field improves the estimator.

Acquire35/100 predictions and complete bills serially after the cost-rule panel
has retired. Score against their stored1e9 means only after acquisition. If both
worsen, close this fixed unrefitted direction; do not tune a coefficient on those
two development networks. Any promising result needs separate fit/check scope
and the established public8/virgin6 panel before deployment.
