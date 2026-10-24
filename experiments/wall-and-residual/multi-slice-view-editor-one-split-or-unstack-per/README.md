# Multi-slice view editor: one split or unstack per operand set, byte-identical, -533 cold / -467 warm requests

2026-10-16T18:09:40Z · experiment · adopted · measured

The multi-slice editor replaces per-chunk getitem views with one fnp.split per operand in the k-split chunk loops, one split of the frame for the six per-cohort views of the u/v pair-mean split, and one unstack for the (lo, hi) pairs of the strided basis views; helpers are appended last and every anchor is asserted. On V-A+B it kept predictions and flops identical while removing requests and residual instructions; it composes on S4 and S5 and was later carried as the MS diet in the audited runtime lineage.

Recorded measurement:

    V-A+B 388cdd4d -> 1f30c16c, normal boundary: requests -533 cold (47,404 -> 46,871) / -467 warm (45,517 -> 45,050), flops +0, pred_35 / pred_100 byte-identical, residual instructions -0.90% (-19.5M) cold / -2.95% (-62.2M) warm; native charged ops +153/+186 (split/unstack counted in-process, not on the platform). Multi-output views cost ~4.4K instructions per extra handle.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_callform_v1/editor_ms_v1.py](code/opus_callform_v1/editor_ms_v1.py)
- [code/opus_callform_v1/build_ms_v1.py](code/opus_callform_v1/build_ms_v1.py)
- [code/opus_callform_v1/cand_vab_ms_v1.py](code/opus_callform_v1/cand_vab_ms_v1.py)
- [code/opus_callform_v1/cand_s5_ms_v1.py](code/opus_callform_v1/cand_s5_ms_v1.py)
- [code/opus_passrel_v1/frontier_v1/cand_frontier_VAB_v1.py](code/opus_passrel_v1/frontier_v1/cand_frontier_VAB_v1.py)
- [code/opus_passrel_v1/stack_v1/cand_s4.py](code/opus_passrel_v1/stack_v1/cand_s4.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4437](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4437)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4446](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4446)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4657](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4657)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-multi-slice-view-editor-one-split-or-unstack-per) · [Copy provenance](../../../PROVENANCE.json)

