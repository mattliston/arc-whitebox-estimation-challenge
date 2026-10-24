# Join core numerics not a lever: float64 solve byte-identical, float64 QR rounding only, Rayleigh-Ritz core +0.87% bill

2026-10-16T14:53:40Z · experiment · rejected · derived

Three changes to the numerics of the live join (the Nystrom core formed by an f32 QR of Y = G[:, rows] and an f32 solve) were tested on the estimator submitted as #336412 (S2tr): the core solve in float64 gave byte-identical predictions (numpy already computes the float32 solve and QR in float64 internally); the QR in float64 changed only rounding for +0.67% bill; and a Rayleigh-Ritz core (storing the represented bank frame Gram instead of the Nystrom core) cost +0.872% bill and moved predictions by 8.7e-6. The join prize lives in the span (which rows and directions), not in the core.

Recorded measurement:

    JS64 (266e83e9): predictions byte-identical to ctl_s2tr on 4/4 seeds, bill +0.12%. JQS64 (289d3f2e): bill +0.67% on seed 1, rounding-level change. JRR (a7002d2d): cold bill +0.872%, predictions moved 8.7e-6.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_ladder2h_v1/build_joinf64_v1.py](code/opus_ladder2h_v1/build_joinf64_v1.py)
- [code/opus_ladder2h_v1/build_jrr_v1.py](code/opus_ladder2h_v1/build_jrr_v1.py)
- [code/opus_ladder2h_v1/cand_s2tr_l2h_js64_v1.py](code/opus_ladder2h_v1/cand_s2tr_l2h_js64_v1.py)
- [code/opus_ladder2h_v1/cand_s2tr_l2h_jqs64_v1.py](code/opus_ladder2h_v1/cand_s2tr_l2h_jqs64_v1.py)
- [code/opus_ladder2h_v1/cand_s2tr_l2h_jrr_v1.py](code/opus_ladder2h_v1/cand_s2tr_l2h_jrr_v1.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4241](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4241)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4244](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4244)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4245](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4245)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4255](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4255)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4256](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4256)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-join-core-numerics-not-lever-float64-solve-byte-identical) · [Copy provenance](../../../PROVENANCE.json)

