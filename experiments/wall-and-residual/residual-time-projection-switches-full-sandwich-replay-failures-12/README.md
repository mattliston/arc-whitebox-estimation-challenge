# Residual-time projection switches to a full sandwich: replay failures 12 to 5 at about 0.04% mean bill

2026-10-16T05:44:44Z · experiment · inconclusive · measured

A local read of the live residual context projected remaining cost at layer boundaries and switched to a request-cheaper exact sandwich when needed. The untriggered path was byte-identical; forced and replay tests quantified relief and cost.

Recorded measurement:

    threshold.39s,L3-L12;forcedL6-2823requests/+1.23%FLOPs,L8-2192/+.94%,L10-1563/+.66%;forcedL6xMSE-.009%/+1.19%bill;replay12to5,trigger~3%,meanbill~+.04%

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L3922](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3922)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3902](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3902)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3918](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3918)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3939](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3939)

<details>
<summary>All result files and records (13)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_census_v1.py.txt](fragment_census_v1.py.txt) · result
- [GRADE_nt6v3_v1.json](GRADE_nt6v3_v1.json) · result
- [GRADE_nt_fe3v4f_v1.json](GRADE_nt_fe3v4f_v1.json) · result
- [GRADE_nt_fl6v4c_v1.json](GRADE_nt_fl6v4c_v1.json) · result
- [GRADE_nt_l6v4c_v1.json](GRADE_nt_l6v4c_v1.json) · result
- [GRADE_nt_shed8v4d_v1.json](GRADE_nt_shed8v4d_v1.json) · result
- [GRADE_psm_v2_v1.json](GRADE_psm_v2_v1.json) · result
- [GRADE_psmf_v2_v1.json](GRADE_psmf_v2_v1.json) · result
- [fragment_governor_v1.py.txt](fragment_governor_v1.py.txt) · code
- [fragment_governor_v2.py.txt](fragment_governor_v2.py.txt) · code
- [fragment_trace_v2.py.txt](fragment_trace_v2.py.txt) · code
- [fragment_trace_v3.py.txt](fragment_trace_v3.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-residual-time-projection-switches-full-sandwich-replay-failures-12) · [Copy provenance](../../../PROVENANCE.json)

