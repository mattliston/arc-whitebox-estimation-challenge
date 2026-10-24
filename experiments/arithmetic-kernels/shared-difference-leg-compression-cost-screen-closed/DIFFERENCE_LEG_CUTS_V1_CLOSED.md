# Shared difference-leg compression: cost screen closed

September 5, 2026. No carrier execution, truncation, target access or fit.
The older per-cohort low-rank P/Q family remains closed (T95..97).
This paper-cost screen considered a different geometry: common legs of
paired signed cubes stay dense; only difference legs share a rank-128
basis, changed at two fixed cuts (8 and 12), retaining four young cohorts.

Existing measured Winograd cost polynomials, current rank35/100 mask/birth
shapes and the public float32 eigendecomposition charge were used. Even
optimistically reusing each old plane for free, while charging basis
acquisition and new repeated-plane restoration, projected total savings
are only 4.000% / 4.357% of those networks' current measured first-context
FLOPs. Raw accuracy is unknown and could deteriorate.

It fails the >=15% cost gate before any accuracy work. No SVD execution,
rank/cut sweep, coadaptation, port, panel or submission is justified for
this proposal. Evidence: difference_leg_cuts_cost_v1.json and the explicit
operation model in price_difference_leg_cuts_v1.py. These are analytical
projections, not a newly measured utilization or benchmark speedup.
