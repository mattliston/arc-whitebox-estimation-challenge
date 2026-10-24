# Gaussian fourth-order term restored in all channels: two-network adjusted +0.07%/+0.57%

2026-10-14T19:51:57Z · experiment · rejected · measured

Restoring the Gaussian fourth-order R4 contraction in all channel pairs (28 R4 calls per network) made both fixed networks worse against the #336075 controls.

Recorded measurement:

    adjusted +0.0654690677% / +0.5691543258%, geometric +0.3169955758%; mean log MSE +0.2570406899%, mean log bill +0.0594535142%; +155,975,680 FLOPs each

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_gaussian_r4_restore_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_gaussian_r4_HELD_v1.py](code/fragment_gaussian_r4_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3034](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3034)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3040](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3040)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3042](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3042)

<details>
<summary>All result files and records (3)</summary>

- [ROOT_RETIREMENT_v1.json](ROOT_RETIREMENT_v1.json) · record
- [SOURCE_READY_HELD_v1.json](SOURCE_READY_HELD_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-gaussian-fourth-order-term-restored-all-channels-two-network) · [Copy provenance](../../../PROVENANCE.json)

