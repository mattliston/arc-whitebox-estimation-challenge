# Positive rank-one cycle component: illustrative product schedule only

September 7, 2026 PDT. Supplement to GAUSSIAN_FOUR_CYCLE_SOURCE_PREFLIGHT_V1.md.
No numerical run, source policy, candidate or score is admitted here.

An illustrative, UNPRESCRIBED schedule takes one CP4 birth after each
layer0..14; emits three cubic atoms per incoming CP4 factor at nonlinear
layers1..14; and uses16 power iterations plus a Rayleigh product per birth.
At pre-layer l the fourth count is l and extra cubic count is3*(l-1)*l/2.

For the OLD scalar-final read convention, classical product counts in
units n^2 are:

| Block | n^2 units |
|---|---:|
| CP4 pushes1..15 |240|
| CP4 K22 and K31 full reads1..14 |420|
| Cubic pushes1..15 |3360|
| Cubic pre-plane full reads1..14 |2730|
| Newly emitted cubic post-plane reads |630|
| Sixteen power iterations for fifteen births |480|
| One final Rayleigh product per birth |30|
| Total |7890|

At n1024 this is8.27326464B classical products. The agent's initial
rough8.2B estimate had not included the final Rayleigh products; preserve
that correction rather than treat8.2B as a complete price. Inherited post
cubic planes are assumed reused through elementwise gating, whose work
is additional. General old scalar-final formula under this illustrative
schedule: n^2*(L-1)*(2*L^2-L+2*K-2), K power iterations.

**This is not the retained response candidate's final reader.** Its P head
owns all16 full-feature layers, including15. A conservative full-final
positive-component price adds60*n^2 for the two CP4 planes and630*n^2
for315 cubic atoms' final plane:8580*n^2=8.99678208B. Actual response CP
rows are a subset; this full-plane count is a conservative structural
example, not measured current-response geometry or a head switch.

Both examples omit elementwise consumers, normalization/selection and
stability policy, complete source/mask/head implementation, FP32 rounding,
packing, runtime and native precision billing. They do not include exact
historical radial-countermetric carry: that alone costs about483.2B
classical products before its cubic descendants. Only the separately
described APPROXIMATE per-layer radial reset avoids that burden, at
unmeasured approximation error.

No current-state supplier value follows from these low positive-product
subtotals. The full-cost and raw-error admission conditions are unchanged.
