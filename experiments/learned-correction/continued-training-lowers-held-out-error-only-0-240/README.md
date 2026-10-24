# Continued training lowers held-out error by only 0.24–0.43%

2026-10-01T21:50Z · experiment · rejected · measured

Continuing to train the learned per-layer correction with Adam (batch 12, reference learning rates) from the bank's parameters was tested as the hypothesis that the estimator is optimisation-limited. Per-network gradients at the parent are noise around zero in every array, the one consistent rank-one direction was used up within 6 updates, and the held-out gain after 18-25 updates was a fraction of a percent; the hypothesis was not supported and the runs were stopped in favour of a refit on a card with new physics (run M).

Recorded measurement:

    pilot A VAL-A after 18 updates: raw -0.36% (se 0.19, 14/20), EMA -0.37% (se 0.17); after 23: raw -0.04%, EMA -0.27%; 60 untouched newer-rank nets: EMA -0.10% (se 0.11) at 18, -0.43% (se 0.15) at 23; run L start (25 updates) on 100 held-out: -0.237% (se 0.109, 55/100); removing the final-layer correction entirely +3.4% on VAL-A; whole learned mean correction worth 13.5% of final-layer MSE; throughput 53-89 s per network gradient (reference 26 s), 100-145 s under load

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/grad_stats.py](code/grad_stats.py)
- [code/candidates/run_L_ema_u00025/candidate.py](code/candidates/run_L_ema_u00025/candidate.py)
- [code/fresh_curve.py](code/fresh_curve.py)
- [code/otr_common.py](code/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L8](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L8)
- [TEAM_LOG_2026-10-01_to_10-22.md#L29](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L29)
- [TEAM_LOG_2026-10-01_to_10-22.md#L30](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L30)
- [TEAM_LOG_2026-10-01_to_10-22.md#L35](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L35)
- [TEAM_LOG_2026-10-01_to_10-22.md#L41](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L41)
- [TEAM_LOG_2026-10-01_to_10-22.md#L47](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L47)
- [TEAM_LOG_2026-10-01_to_10-22.md#L51](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L51)
- [TEAM_LOG_2026-10-01_to_10-22.md#L94](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L94)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.md](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md) · result
- [progress.txt](progress.txt) · log
- [progress-e7ed5e2a.txt](progress-e7ed5e2a.txt) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-continued-training-lowers-held-out-error-only-0-240) · [Copy provenance](../../../PROVENANCE.json)

