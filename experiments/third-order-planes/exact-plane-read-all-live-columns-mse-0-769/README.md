# Exact plane read on all live columns: MSE -0.769% for +3.236% bill

2026-10-16T14:02:08Z · experiment · inconclusive · measured

Removing both the prefix cut and rank-two tail approximation exposed a raw-accuracy carrier but was too expensive as a whole. Partial variants were still running at the slice boundary.

Recorded measurement:

    MSE-.769%,SE.85,5/8;bill+3.236%;net+2.467%;maxpredictiondifference6-11e-5

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_ladder2h_v1/census_incl_v1.py](code/lead/runs/opus_ladder2h_v1/census_incl_v1.py)
- [code/lead/runs/opus_ladder2h_v1/build_joinf64_v1.py](code/lead/runs/opus_ladder2h_v1/build_joinf64_v1.py)
- [code/lead/runs/opus_ladder2h_v1/build_joinf64_v2.py](code/lead/runs/opus_ladder2h_v1/build_joinf64_v2.py)
- [code/lead/runs/opus_ladder2h_v1/build_jrr_v1.py](code/lead/runs/opus_ladder2h_v1/build_jrr_v1.py)
- [code/lead/runs/opus_ladder2h_v1/build_ladder2h_v1.py](code/lead/runs/opus_ladder2h_v1/build_ladder2h_v1.py)
- [code/lead/runs/opus_ladder2h_v1/build_ladder2h_v2.py](code/lead/runs/opus_ladder2h_v1/build_ladder2h_v2.py)
- [code/lead/runs/opus_ladder2h_v1/build_ladder2h_v3.py](code/lead/runs/opus_ladder2h_v1/build_ladder2h_v3.py)
- [code/lead/runs/opus_ladder2h_v1/build_ladder2h_v4.py](code/lead/runs/opus_ladder2h_v1/build_ladder2h_v4.py)
- [code/lead/runs/opus_ladder2h_v1/build_ladder2h_v5.py](code/lead/runs/opus_ladder2h_v1/build_ladder2h_v5.py)
- [code/lead/runs/opus_ladder2h_v1/build_layerr_v1.py](code/lead/runs/opus_ladder2h_v1/build_layerr_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/opus_ladder2h_v1/cand_s2tr_l2h_jrr_v1.py](code/lead/runs/opus_ladder2h_v1/cand_s2tr_l2h_jrr_v1.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py](code/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4192](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4192)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [census_s2tr_v1.json](../../covariance-structure/exact-learned-covariance-edge-mse-0-035-0-751/census_s2tr_v1.json) · result
- [guard_census_s2tr.json](../../covariance-structure/exact-learned-covariance-edge-mse-0-035-0-751/guard_census_s2tr.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-exact-plane-read-all-live-columns-mse-0-769) · [Copy provenance](../../../PROVENANCE.json)

