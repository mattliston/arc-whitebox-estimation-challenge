# Skew-corrected birth density lowers native MSE 0.34% on ten networks for 105 extra requests

2026-10-02T20:06Z · experiment · inconclusive · measured

Multiplying birth weights by the skew-corrected kink density lowers native final MSE 0.34% against two arithmetic draws of submission #333607, with eight of ten networks improved. The effect matches the research implementation but is only about one standard error on this sample.

Recorded measurement:

    13.9967e-9 vs14.0443e-9; pooled -0.34%; mean per-network -0.36% (se0.32),8/10; each reference draw -0.29% /-0.39%; network328 +2.05%; +105 requests, +0.002% bill; second reference draw +0.10% MSE, +0.078% bill, -410 requests

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/a41_base/fragment_bsk_v1.py](code/a41_base/fragment_bsk_v1.py)
- [code/a42_head/judge_ten.py](code/a42_head/judge_ten.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L463](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L463)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-skew-corrected-birth-density-lowers-native-mse-0-34) · [Copy provenance](../../../PROVENANCE.json)

