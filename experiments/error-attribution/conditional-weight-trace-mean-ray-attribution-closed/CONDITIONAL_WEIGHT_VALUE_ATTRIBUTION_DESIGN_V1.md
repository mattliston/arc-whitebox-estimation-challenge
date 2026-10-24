# Which covariance statistics drive next-layer mean error?

Phase2 only, September5,2026,21:23PDT. No fitting, new truth, or carrier.

For isotropic Gaussian next weights w and fixed current mean m, write
u=m/||m||, z=w.u and w_perp=w-z*u. Conditional on z and ||w_perp||,
the exact mean quadratic action of a fixed covariance error E is

    (u^T E u)*z^2 + (tr(E)-u^T E u)/(n-1)*||w_perp||^2.

This isolates two scalar invariants without choosing coefficients from
the observed output vector. Multiply by the same saved next-mean response
used by the source-v2 diagnostic. That response also depends on w, so the
resulting product is an explicitly approximate value decomposition, not
an exact conditional-expectation claim for the weighted output.

Use all existing rank35 source cells8..14 with their original coherent8M
covariance labels and frozen host means. Report natural two-statistic,
trace-only and mean-ray contributions, and their unfit squared-error
ratios. Do not select a layer or an arm for an estimator. A mean-ray
consumer-slice research design is justified only if the two-statistic
action removes at least50% pooled next-mean error and its mean-ray term
has at least25% of the baseline value energy. Otherwise no new q1 MC or
basis acquisition follows. These are oracle attribution criteria only.
