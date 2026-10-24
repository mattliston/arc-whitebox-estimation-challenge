# Moving the unsquared query leg of Q = W^T P keeps two products and loses 302272 r FLOPs on the old 832-864 geometry

2026-09-08T07:19Z · finding · rejected · derived

Rewriting the plane read as K_IJ = (Q_I^2 Lambda P^T) W_J, i.e. applying the unsquared query leg after the weighted factor product, retains two products instead of one while the full Q is still needed for marginal cubes and state. The ordinary-product saving r [(c-a)(2R-1) - c(2a-1)] is negative whenever support shrinks and is already negative on the old a832 to c864, R17728 expansion, so no prototype was built.

Recorded measurement:

    saving on the OLD a832 -> c864, R17728 geometry: -302272*r FLOPs

## Evidence

- [SOURCE_DECISIONS_2026_09_08_V1.md#L195](SOURCE_DECISIONS_2026_09_08_V1.md#L195)
- [SOURCE_DECISIONS_2026_09_08_V1.md#L200](SOURCE_DECISIONS_2026_09_08_V1.md#L200)

<details>
<summary>All result files and records (1)</summary>

- [SOURCE_DECISIONS_2026_09_08_V1.md](SOURCE_DECISIONS_2026_09_08_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps) · [Copy provenance](../../../PROVENANCE.json)

