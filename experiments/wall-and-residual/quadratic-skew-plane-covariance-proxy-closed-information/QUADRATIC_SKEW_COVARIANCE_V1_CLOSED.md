# Quadratic skew-plane covariance proxy: closed on information

September5,2026, actual clock. Fixed design in
`QUADRATIC_SKEW_COVARIANCE_V1.md`; original thresholds unchanged.

The n4 exact Gaussian polynomial/conditional-projection identities pass
below7e-17. The analytic covariance response agrees with a finite difference
of the actual parent map at relative9.692e-11. No exact global fourth
tensor is claimed for the pairwise proxy.

Cost v1 projected12.88% extra. Source audit found that FlopScope.12 ignores
the symmetric flag for solve_cost: it charges generic LU. The v1 number
already charged that LU cost, but a literal positive-definiteness guard
also needs a Cholesky. Version2 includes BOTH and projects13.448% extra
on35/100, still within the original15% gate. Neither is a native bill.

The six training rows and two heldouts all reproduce the old coherent8M
input fields and covariance targets BYTE EXACTLY. Features and input
fingerprints are frozen before current-target feedback. The fixed3-column
ridge and2-column control are frozen before heldout rows are loaded.

Heldout400: covariance ratio .995716259902, next-mean response .993161296841.
Heldout500: covariance ratio .996535640480, next-mean response1.000413621442.
New/control next-mean ratios .999834525933 /1.000066289726. All original
25% and incremental10% gates fail. The normalized fit is well conditioned
(condition2.745), not an optimizer issue. No larger fit, proxy expansion,
rollout, panel, candidate, native port or submission follows this result.

Job64.421s, peakRSS6,273,368,064bytes. All workers complete. Receipts:
`quadratic_skew_covariance_certificate_v1.json`, cost_v1/v2 JSONs,
`quadratic_skew_covariance_verdict_v1.json`, and
`quadratic_skew_covariance_progress_v1/`. The model NPZ is diagnostic only.
Official usable #329926 remains adjusted4.982231460183304e-9.
