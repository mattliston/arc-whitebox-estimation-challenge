# Per-unit kurtosis is 8-11% low by median but 1.5-3.3% low after mean-formula weighting

2026-10-11T18:50:46Z · finding · rejected · measured

The S47e moment-closure estimator's own pair-map k4 input (including cohort-age local fourth-order term (K4L) and F4C, the fourth-order terms) was compared with the analyst's 32M-sample per-unit truths on network 35 at layers 12-14: median excess kurtosis k4 / v^2 moment-closure estimator 0.049 / 0.051 / 0.053 vs truth 0.053 / 0.057 / 0.060, but the shortfall sits far from the kink (|alpha| > 3, where k4 barely moves the mean); weighted as the mean formula uses k4 ((alpha^2 - 1) pdf / (24 v s)) the whole-layer scale moment-closure estimator -> truth is 1.020 / 1.033 / 1.015, and per unit the moment-closure estimator tracks the truth with correlation only 0.58-0.84 per band. The k4 error is unit-level scatter, not a per-layer level; a per-layer literal toward the truth fights the co-adapted band dial (0.96 at 12-14) and is worth about 0.1% at most (the kurtosis channel is ~4% of the final error).

Recorded measurement:

    median k4 / v^2 card 0.049 / 0.051 / 0.053 vs truth 0.053 / 0.057 / 0.060 (card ~8-11% low); |alpha| > 3 card / truth 0.82-0.85 (scale 1.20-1.26); weighted whole-layer scale 1.020 / 1.033 / 1.015; |alpha| < 2 bands 0.99-1.07; per-unit correlation 0.58-0.84 per band (truth noise 0.8-0.9% of |k4|); refit's k4 band dial 12-14 = 0.96; worth ~0.1% at most

## Programs

- [t114_k4_probe.py](t114_k4_probe.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1161](../../TEAM_LOG_2026-10-01_to_10-22.md#L1161)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-11-per-unit-kurtosis-8-11-low-median-but-1) · [Copy provenance](../../../PROVENANCE.json)

