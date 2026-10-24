# Per-layer dial gradients favor a smaller birth-source scale but mostly reflect boundary layers

2026-10-02T01:28Z · finding · informational · measured

Sixteen per-layer dial families (217 scalars) were certified as zero vectors added to the global dials and their exact gradients recorded free with every training gradient. Over 72 networks 39 scalars were beyond 2 standard errors (noise: 10); the largest item was global, the b-carrier source scale wanting s < 1 (forward scan parabola minimum s 0.85, -0.67%); k4, iso, k21n, edge, kappa, X0, X2 and s_carry were noise. Later signals over 38 networks (s_carry up at layer 14 and down at 1-4, bscale down at 7-14 and up at 0-3, the final layer's k4 down) were end effects; a depth profile of s_carry was worth -0.11 to -0.13%, not a native table.

Recorded measurement:

    217 scalars; four networks: 20 unanimous signs (chance 27); 72 networks: 39 beyond 2 se (noise 10), 11 beyond 3 (noise 0.6); bsrc t +4.3, scan 0.9 / 0.8 / 0.7: -0.60% / -0.59% / -0.40%, minimum 0.85 -0.67%; recentring t +4.9 but -0.18% per -0.1; 38 networks: 22 of 300 per-layer scalars beyond 3 se (0.8 by chance); s_carry profile -0.11% (se 0.08) / -0.13% (se 0.16)

## Programs

- [../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py](../../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/trainer/otr_pl.py)
- [pl_stats.py](pl_stats.py)
- [pl_scan.py](pl_scan.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L50](../../TEAM_LOG_2026-10-01_to_10-22.md#L50)
- [TEAM_LOG_2026-10-01_to_10-22.md#L70](../../TEAM_LOG_2026-10-01_to_10-22.md#L70)
- [TEAM_LOG_2026-10-01_to_10-22.md#L169](../../TEAM_LOG_2026-10-01_to_10-22.md#L169)
- [TEAM_LOG_2026-10-01_to_10-22.md#L204](../../TEAM_LOG_2026-10-01_to_10-22.md#L204)
- [RESULT.md#L286](../2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md#L286)

<details>
<summary>All result files and records (3)</summary>

- [pl_scan_bsrc_u6.json](pl_scan_bsrc_u6.json) · result
- [RESULT.md](../2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-per-layer-dial-gradients-favor-smaller-birth-source-scale) · [Copy provenance](../../../PROVENANCE.json)

