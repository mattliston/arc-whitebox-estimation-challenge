# Old top-8 subspace-integration capture of 6.526% was a first-order ANOVA share, not variance removed by integration

2026-09-08T07:19Z · finding · informational · derived

Re-reading measure_realnet.py showed that the August 25 P3 experiment fixed the top subspace A and sampled its complement B, so its reported capture V_A/V is a first-order ANOVA share; perfectly integrating A while sampling B would remove V_A + V_AB, not V_A alone. Gaussian conditional Poincare plug-ins from the old sampled gradient Gram give (V_A+V_AB)/V <= .214248775 for top2 and .754954615 for top8, which are empirical quantities, not certified bounds, and the executed two-direction Gauss-Hermite grid stays closed.

Recorded measurement:

    old top8 reported capture 6.526%; Poincare plug-ins top2 .214248775, top8 .754954615 (160 sampled Jacobians); executed two-direction GH grid MSE .0010106713 at 4096 forwards, about 53.4x matched Monte Carlo

## Evidence

- [SOURCE_DECISIONS_2026_09_08_V1.md#L160](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md#L160)
- [SOURCE_DECISIONS_2026_09_08_V1.md#L169](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md#L169)
- [SOURCE_DECISIONS_2026_09_08_V1.md#L176](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md#L176)
- [SOURCE_DECISIONS_2026_09_08_V1.md#L181](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md#L181)

<details>
<summary>All result files and records (1)</summary>

- [SOURCE_DECISIONS_2026_09_08_V1.md](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-old-top-8-subspace-integration-capture-6-526-first) · [Copy provenance](../../../PROVENANCE.json)

