# Stacking elementwise operations in the moment core saved no requests after stack and unstack costs

2026-10-17T17:03:31Z · experiment · rejected · measured

The per-unit moment core contained many small-vector operations, but they formed dependent chains of distinct operators. Every candidate stack group required four requests, eliminating the proposed savings; only approximately thirty exact cast-folding requests remained, so no fragment was shipped.

Recorded measurement:

    2,755 charged operations; 1,864 vector operations; 5 same-operation runs per layer of lengths 4, 3, 5, 3, 4; perfect stacking saves 0 (-1 net); ~30 requests per network versus a 600 target.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_vec_v1/probe_fhm_dag_v1.py](code/lead/runs/opus_vec_v1/probe_fhm_dag_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4892](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4892)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [fhm_ops_7103256.json](fhm_ops_7103256.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-stacking-elementwise-operations-moment-core-saved-no-requests-after) · [Copy provenance](../../../PROVENANCE.json)

