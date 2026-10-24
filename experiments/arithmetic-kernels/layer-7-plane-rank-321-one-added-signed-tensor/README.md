# Layer-7 plane rank 321 (one added signed tensor-trace direction): adjusted +6.18%, the odd rank disables aligned kernels

2026-10-14T21:34:47Z · experiment · rejected · derived

Adding one orthogonal omitted signed tensor-trace direction to the layer-7 rank-320 basis (rank 321) raised the bill by 3.2% because the non-64-aligned rank disabled the aligned fast paths (RLF products 8 to 5, UBX_RC products 3 to 0); the local 120M-FLOP estimate was not a whole-bill bound. Closed on the first network.

Recorded measurement:

    7100136: adjusted +6.18444205%, log MSE +2.79430608%, log bill +3.20643546%, +8,312,076,952 FLOPs

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_l7_trace321_rescue_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_l7_trace321_HELD_v1.py](code/fragment_l7_trace321_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3110](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3110)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3112](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3112)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3116](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3116)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3118](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3118)

<details>
<summary>All result files and records (2)</summary>

- [CLOSED_COST_MISS_SOURCE_NOTE_v1.json](CLOSED_COST_MISS_SOURCE_NOTE_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Layer-7 rank-384 one-coordinate swap for a signed trace direction: two-network adjusted +0.44% (MSE +0.41%), closed](../../third-order-planes/layer-7-rank-384-one-coordinate-swap-signed-trace/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-layer-7-plane-rank-321-one-added-signed-tensor) · [Copy provenance](../../../PROVENANCE.json)

