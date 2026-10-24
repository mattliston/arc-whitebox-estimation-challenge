# Layer-7 rank-384 one-coordinate swap for a signed trace direction: two-network adjusted +0.44% (MSE +0.41%), closed

2026-10-14T21:38:57Z · experiment · rejected · measured

Swapping the minimum weighted-Gram-energy coordinate of the layer-7 rank-384 basis for the signed trace direction (keeping all aligned ranks) kept the fast paths but lost accuracy on average.

Recorded measurement:

    paired adjusted +0.436740951% (7100136 -0.3460183%, 7100137 +1.2256486%); mean log MSE +0.409637188%, log bill +0.026152817%; +68,244,371 FLOPs each

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_l7_trace384_swap_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_l7_trace384_swap_HELD_v1.py](code/fragment_l7_trace384_swap_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3120](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3120)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3122](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3122)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3126](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3126)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3128](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3128)

<details>
<summary>All result files and records (2)</summary>

- [SOURCE_READY_HELD_v1.json](SOURCE_READY_HELD_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Layer-7 plane rank 321 (one added signed tensor-trace direction): adjusted +6.18%, the odd rank disables aligned kernels](../../arithmetic-kernels/layer-7-plane-rank-321-one-added-signed-tensor/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-layer-7-rank-384-one-coordinate-swap-signed-trace) · [Copy provenance](../../../PROVENANCE.json)

