# Output-symmetry compiler for the triangle part of the full-prefix query

September6,2026. The earlier full query schedule closed an optimistic
59.78%/59.01% added-cost subtotal versus a50% prerequisite. Its kernel
schedule computed full squares for known symmetric triangle contractions.
This is a different matrix-product compiler, not new source/age selection,
rank, gain, target, fit or a reopening of the same old schedule.

For symmetric physical covariance C with diagonal removed and a query h,
the triangle slice is diag(d)[C elementwise (C diag(d*h) C)]diag(d).
Both the inner weighted Gram and the final A.T slice A are symmetric.
Only the first multiplication slice A is a general rectangular product.
The small asymmetry of existing f32 covariance is a NUMERICAL caveat:
an executable version must explicitly symmetrize its triangle-only C and
certify the resulting discrepancy against the old source before a wide
signal claim. No bank covariance is silently symmetrized.

First price an optimistic scalar-FLOP lower workload. For a known-symmetric
product with output m x m and inner dimension k, allow any dyadic (or other
32-aligned) output split s, computing the upper off-diagonal rectangle once
and the two diagonal symmetric products recursively. Dynamic programming
chooses min(full existing matmul, sym(s,k)+sym(m-s,k)+mm(s,k,m-s)). All
leaves use the same existing plain/Winograd3/4/5 prices. No empirical data
or target enters this compiler optimization. Also report matrix-call count
and depth of the minimum-cost tree; no runtime saving is assumed.

Omit the IDENTICALLY ZERO mixed source at L0: mu0=0 implies eta=e=0,
and input k3/k4=0 implies D=0, hence F=0. This is algebraic dead-code
elimination, not source selection by measured score. Keep its nonzero
Gaussian triangle and keep all other13-source physics and q4 triangle rank.

Use the same35/100 active-shape tapes and original50% added subtotal gate.
Keep the same omitted costs explicitly FREE, so a passing result is only
a prerequisite for complete costing/runtime and source-fidelity work.
If even this optimistic compiler fails, do not implement a wide triangle
kernel or accuracy run. Do not claim a candidate bill or score.
