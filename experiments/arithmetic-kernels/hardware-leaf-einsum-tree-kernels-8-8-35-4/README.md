# Hardware leaf einsum for the tree kernels: +8.8-35.4% kernel time and +2.4-11.1% leaf bill against matmul, closed

2026-10-17T01:04:40Z · experiment · rejected · measured

Replacing the leaf matmuls of the production tree kernels by einsum on five actual P10/P20 leaf shapes with production strides gave byte-equal outputs but slower kernels, more server instructions and a higher leaf bill (einsum bills 2k against 2k-1), so the route was closed without integration.

Recorded measurement:

    30 alternating matmul/einsum operations: einsum +8.8-35.4% kernel, +2.1-18.8% server instructions, +2.44-11.11% leaf bill; repaired prefix run peak 1.682 GB.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_prefix128_fusion_v1/candidate.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/einsum_leaf_v1/client_leaf_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/einsum_leaf_v1/client_leaf_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/einsum_leaf_v1/run_leaf_v2.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/einsum_leaf_v1/run_leaf_v2.py)
- [code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_cap750_w3_zb4_runtime_v8.py](code/opus_reexam_20261001_v1/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_cap750_w3_zb4_runtime_v8.py)
- [code/opus_reexam_20261001_v1/release/common.py](code/opus_reexam_20261001_v1/release/common.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py](code/opus_reexam_20261001_v1/release/variants/s42_0fccf396_l3b128/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4636](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4636)

<details>
<summary>All result files and records (2)</summary>

- [RESULT_v2.json](RESULT_v2.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-hardware-leaf-einsum-tree-kernels-8-8-35-4) · [Copy provenance](../../../PROVENANCE.json)

