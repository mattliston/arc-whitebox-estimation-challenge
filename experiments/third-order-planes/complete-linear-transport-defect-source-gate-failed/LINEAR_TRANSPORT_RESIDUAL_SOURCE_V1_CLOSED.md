# Complete linear transport defect: source gate failed

Phase2, September5,2026. Certificates below1.1e-17; current host
byte-identical. Fixed rank35 post12 -> pre13, no new Monte Carlo.

The full star-plus-metric-reset source gives consumer-weighted signal
error ratio1.02147212 and cosine-.04779461. Its q4 residual projection
gives1.02530197 and cosine-.11507238; independent ratios1.02591541 /
1.02344294. Unweighted ratios improve modestly (.96547/.96751), but
the relevant weighted information gate FAILS. Source slice norm.16777,
versus about1.90 for the independently measured full residual slices.

No recursive sidecar, all-star refit, damping, term-subset selection or
bank follows. projected_retained_full_gain in the JSON is mathematically
inapplicable because the full arm worsens weighted error; its large
negative sentinel-like value is not an informative ratio. The gate does
not use it when the full arm worsens. Preserve the original receipt.
Evidence: linear_transport_residual_source_r35_l12_v1.json/.npz.
