# Float64 work is only 0.25% of metered cost across about 40 sites

2026-10-19T13:25:26Z · experiment · rejected · measured

A 0.12.3-weighted census of the base on net 7103256 puts the charges touching float64 at 0.626 GF pre-weight, 0.25% of the bill (the x2 rate makes the surcharge 0.25% of the weighted bill), spread over about 40 small sites with the largest at 0.025%: converting them to float32 cannot pay for its numerical risk.

Recorded measurement:

    0.626 GF pre-weight = 0.25% of the bill; ~40 sites, largest 0.025% (_ps32 asarray, fhm core matmuls, gop2 contract, features)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/fable_oct19_audit_v1/probe_charges_v2.py](code/lead/runs/fable_oct19_audit_v1/probe_charges_v2.py)
- [code/lead/runs/fable_oct19_fusion_pb_v1/candidate_fusion_pbv1_v2.py](code/lead/runs/fable_oct19_fusion_pb_v1/candidate_fusion_pbv1_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5952](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5952)

<details>
<summary>All result files and records (2)</summary>

- [charges_v123_a7ffd713_7103256.json](../../../records/research/2026-10-19-flopscope-version-audit-metered-paths-all-0-12-3/charges_v123_a7ffd713_7103256.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-float64-work-only-0-25-metered-cost-across-about) · [Copy provenance](../../../PROVENANCE.json)

