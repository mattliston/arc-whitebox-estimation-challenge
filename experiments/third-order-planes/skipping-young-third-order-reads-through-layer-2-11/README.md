# Skipping young third-order reads through layer 2: 11.6-12x MSE for 2.4-2.6% fewer FLOPs

2026-10-20T15:04:17Z · experiment · rejected · measured

Removing young third-order plane reads at the first layers caused severe accuracy loss on both fresh networks. Larger layer ranges were abandoned after the first setting.

Recorded measurement:

    - [fable-lead] 2026-10-20T15:04:17Z NOREAD DEAD (candidate_noread_sci_v1.py 1fa1915a: skip the young (2,1) plane read _cp_read_young_sq for layers <= L; fresh 7100005/7100007): L<=2 -> U 1.63e-7 / 1.50e-7 (x12 / x11.6 the base) for bill -2.6% / -2.4%; the early third-order plane content is essential at the first layers (not merely co-adapted), the 20%-of-bill early reads cannot be dropped. Evals killed after the L<=2 setting; the L<=4/6 settings are moot.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_noread_sci_v1.py](code/candidate_noread_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6190](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6190)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-skipping-young-third-order-reads-through-layer-2-11) · [Copy provenance](../../../PROVENANCE.json)

