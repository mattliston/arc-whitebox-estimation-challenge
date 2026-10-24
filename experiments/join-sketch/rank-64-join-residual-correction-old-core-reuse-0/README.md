# Rank-64 join residual correction with old-core reuse: +0.351% adjusted on two native networks

2026-10-18T04:18:11Z · experiment · rejected · measured

A rank-64 correction folded a saved join residual into the existing read, combined with reuse of an old core product. The native pair increased both raw error and bill, so it was stopped without extension.

Recorded measurement:

    Raw +.097%, bill +.253%, adjusted +.351%; core-product reuse alone: bill -.095%, raw +.054%, net -.041%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct18_accuracy_v1/build_join_left_repair_v1.py](code/astra_oct18_accuracy_v1/build_join_left_repair_v1.py)
- [code/astra_oct18_accuracy_v1/candidate_jlr_residual64_qcore_HELD_v1.py](code/astra_oct18_accuracy_v1/candidate_jlr_residual64_qcore_HELD_v1.py)
- [code/astra_oct18_accuracy_v1/fragment_join_left_repair_v1.py](code/astra_oct18_accuracy_v1/fragment_join_left_repair_v1.py)
- [code/astra_oct18_accuracy_v1/probe_l13_row_repair_v1.py](code/astra_oct18_accuracy_v1/probe_l13_row_repair_v1.py)
- [code/astra_oct18_accuracy_v1/tiny_jlr_ownership_v1.py](code/astra_oct18_accuracy_v1/tiny_jlr_ownership_v1.py)
- [code/opus_fix336687_v1/candidate_v61j.py](code/opus_fix336687_v1/candidate_v61j.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5171](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5171)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5179](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5179)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [JLR64_NATIVE_CLOSED_v1.json](JLR64_NATIVE_CLOSED_v1.json) · result
- [GRADE_STORE_v1.json](GRADE_STORE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-rank-64-join-residual-correction-old-core-reuse-0) · [Copy provenance](../../../PROVENANCE.json)

