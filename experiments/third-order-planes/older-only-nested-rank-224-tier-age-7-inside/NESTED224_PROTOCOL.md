# Fixed older-only nested tier

This is the previously unexecuted age7/rank224 construction documented in
`current_comp_nested_old224_v1/INDEPENDENT_40_PRICE_AUDIT_20260913_UTC.md`,
now based on the submitted Cost150 source. Keep original age4/rank384,
lambda-squared metric, all signed pairs, all scalar/full-row consumers,
all learned values, masks, and first eight output layers. No rank/age search.

At layers8..14, move exactly one oldest cohort into a rank224 map inside
the rank384 coordinates, using two weighted-Gram rangefinder QR passes.
Update the full core by subtracting the original mover/older core and adding
its projected core. Rotate both the middle factors and nested map at each
first-tier join. Reconstruct every original old column and lift its read
through the nested map. No new projection at the final layer.

First verify core ownership and repeated-plane reads against explicit
dense signed cubes on a small synthetic two-move case. This is an algebra
check, not accuracy evidence. Then run full native35 and100, plus35 repeat,
with all outputs and bills saved before truth scoring. First eight output
layers must match the Cost150 controls byte-for-byte, and repeats must be exact.
Require each network's actual adjusted objective to improve before expansion.
A failure is preserved without tuning. The fixed public8+virgin6 panel and
production boundary remain necessary before any release.

One process group, nice10/BLAS1, hard8GB RSS, maximum256MiB added system swap,
240s guard for three native calls; no new truth acquisition. This is a
modest efficiency component, with no forecast of a sub3e-9 or1e-9 score.

Before any truth scoring: add a fresh cold100 call so both primary objectives
include their own measured initialization. The warm100 call from the repeat
sequence remains a runtime/reproducibility check, not a cold-cost substitute.
