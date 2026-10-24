# Two-stage residual and wall relief: replay leaves zero to one failures but forced approximation raises MSE

2026-10-16T06:20:57Z · experiment · inconclusive · measured

Adding learned-work removal and a last-resort young-read omission reduced simulated failures, but their triggered-network errors were large. This motivated exact request relief before approximate routing.

Recorded measurement:

    full+leanL6-7841req/-17.2%;lean+shedL8-8949/-19.4%FLOPs;forcedleanMSE+36%,shedx39;17failuresleft0/1/1atswitchL3/L6/L8;trigger2.4%/1.0%/.45%/.09%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_governor_v1/grade_v1.py](code/opus_governor_v1/grade_v1.py)
- [code/opus_governor_v1/test_census1.py](code/opus_governor_v1/test_census1.py)
- [code/opus_governor_v1/build_v1.py](code/opus_governor_v1/build_v1.py)
- [code/opus_governor_v1/build_v2.py](code/opus_governor_v1/build_v2.py)
- [code/opus_governor_v1/candidate_GOV_v1.py](code/opus_governor_v1/candidate_GOV_v1.py)
- [code/opus_governor_v1/candidate_GOV_v2.py](code/opus_governor_v1/candidate_GOV_v2.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3952](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3952)

<details>
<summary>All result files and records (13)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_census_v1.py.txt](../residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_census_v1.py.txt) · result
- [GRADE_nt6v3_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt6v3_v1.json) · result
- [GRADE_nt_fe3v4f_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_fe3v4f_v1.json) · result
- [GRADE_nt_fl6v4c_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_fl6v4c_v1.json) · result
- [GRADE_nt_l6v4c_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_l6v4c_v1.json) · result
- [GRADE_nt_shed8v4d_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_shed8v4d_v1.json) · result
- [GRADE_psm_v2_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_psm_v2_v1.json) · result
- [GRADE_psmf_v2_v1.json](../residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_psmf_v2_v1.json) · result
- [fragment_governor_v1.py.txt](../residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_governor_v1.py.txt) · code
- [fragment_governor_v2.py.txt](../residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_governor_v2.py.txt) · code
- [fragment_trace_v2.py.txt](../residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_trace_v2.py.txt) · code
- [fragment_trace_v3.py.txt](../residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_trace_v3.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-two-stage-residual-wall-relief-replay-leaves-zero-one) · [Copy provenance](../../../PROVENANCE.json)

