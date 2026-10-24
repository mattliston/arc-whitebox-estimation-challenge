# Covariance-update broadcast-view removal saved 133–134 requests for a 0.04588% bill cost

2026-10-18T06:21:34Z · experiment · informational · measured

An exact broadcast-view patch eliminated covariance-update view requests. Native SDK tiny checks preserved numerical values and output delta bytes while some intermediate signed zeros differed; the component was held because its copy cost was not needed by a qualifying larger saving.

Recorded measurement:

    Estimated 133/134 fewer requests, +.04588% bill; six actual SDK cases had the same values and delta bytes.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct18_accuracy_v1/build_bd_view_only_v1.py](code/astra_oct18_accuracy_v1/build_bd_view_only_v1.py)
- [code/astra_oct18_accuracy_v1/fragment_bd_view_diet_v1.py](code/astra_oct18_accuracy_v1/fragment_bd_view_diet_v1.py)
- [code/astra_oct18_accuracy_v1/tiny_bd_view_diet_v1.py](code/astra_oct18_accuracy_v1/tiny_bd_view_diet_v1.py)
- [code/astra_oct18_bill_v1/candidate_v61j_bookkeeping_union_HELD_v1.py](code/astra_oct18_bill_v1/candidate_v61j_bookkeeping_union_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5297](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5297)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-covariance-update-broadcast-view-removal-saved-133134-requests-0) · [Copy provenance](../../../PROVENANCE.json)

