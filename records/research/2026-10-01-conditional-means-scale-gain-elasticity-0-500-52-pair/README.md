# Conditional means scale with gain elasticity 0.50–0.52 and pair covariances with 0.91–1.02

2026-10-01T19:00Z · finding · informational · measured

Monte Carlo on network 35 (2M samples, 192 units at layers 6, 10, 14) measured how conditional moments respond to the squared norm of an earlier layer. Means have elasticity 0.50-0.52 (rigid scaling predicts 0.5); per-unit variances 0.81-0.83 against the previous layer's norm rising to 1.00 against the input norm; off-diagonal covariances 0.91-1.02. The collective gain mode is therefore close to a rigid scale mixture for second moments and the card's Sym(C x b) form has the right structure, so the gain mode is not where the error is.

Recorded measurement:

    means 0.50-0.52; per-unit variances 0.81-0.83 (previous layer), 0.87-0.96 (six layers back), 1.00 (input norm); off-diagonal covariances 0.91-0.94 / 0.94-0.98 / 1.00-1.02; dv vs -2 mu dmu correlation -0.05..+0.18 at layers 4-15

## Programs

- [e6_gain_elasticity_mc.py](e6_gain_elasticity_mc.py)

## Evidence

- [SESSION_NOTES_2026-10-01_to_10-03.md#L109](../../SESSION_NOTES_2026-10-01_to_10-03.md#L109)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L115](../../SESSION_NOTES_2026-10-01_to_10-03.md#L115)

<details>
<summary>All result files and records (2)</summary>

- [e6_r35.log](e6_r35.log) · log
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-conditional-means-scale-gain-elasticity-0-500-52-pair) · [Copy provenance](../../../PROVENANCE.json)

