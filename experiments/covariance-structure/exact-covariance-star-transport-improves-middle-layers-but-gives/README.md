# Exact covariance-star transport improves middle layers but gives final MSE +0.159% at +22.3% bill

2026-10-16T14:07:57Z · experiment · rejected · measured

Restoring the full second star transport term improved intermediate means without a final-layer benefit. The result closed partial refinements because late-born error remained untouched.

Recorded measurement:

    finalMSE+.159%,SE.54,3/8;bill+22.3%;L8-1.51%,L9-1.20%,L10-1.26%,L13-1.12%,L14-.17%,L15+.16%

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L4200](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4200)

<details>
<summary>All result files and records (3)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [census_s2tr_v1.json](../exact-learned-covariance-edge-mse-0-035-0-751/census_s2tr_v1.json) · result
- [guard_census_s2tr.json](../exact-learned-covariance-edge-mse-0-035-0-751/guard_census_s2tr.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-exact-covariance-star-transport-improves-middle-layers-but-gives) · [Copy provenance](../../../PROVENANCE.json)

