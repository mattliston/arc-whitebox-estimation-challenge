# Interpolative old-tier representation: three to 3.8 times the lost join energy despite 2.5% bill saving

2026-10-16T13:44:16Z · experiment · rejected · measured

Replacing orthogonal projection with direct Nystrom row interpolation removed QR and frame products but lost too much energy and introduced ill-conditioned float32 solves.

Recorded measurement:

    lostenergy3.0-3.8x;Frobenius1.5-1.7x;condition5e4-2e6;saving~2.5%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_pass_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/interp_offline_v1.py](code/interp_offline_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4173](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4173)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-interpolative-old-tier-representation-three-3-8-times-lost) · [Copy provenance](../../../PROVENANCE.json)

