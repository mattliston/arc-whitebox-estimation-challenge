# Learned-operator products through tree kernels: 0.082% saving canceled by extra runtime-relief triggers

2026-10-16T06:55:48Z · experiment · rejected · measured

Tree routing of small learned products saved modest arithmetic but added enough requests to increase runtime-switch costs and late-detection failures.

Recorded measurement:

    depth3FLOPs-.082%,ops+1780;depth4-.094%/+2235;depth2-.060%/+1325;messages-.099%/+4042;warm+1740/+3.81%;triggerbill+.093%;net+.011%

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

- [TEAM_LOG_2026-10-01_to_10-22.md#L3973](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3973)

<details>
<summary>All result files and records (13)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [fragment_census_v1.py.txt](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_census_v1.py.txt) · result
- [GRADE_nt6v3_v1.json](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt6v3_v1.json) · result
- [GRADE_nt_fe3v4f_v1.json](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_fe3v4f_v1.json) · result
- [GRADE_nt_fl6v4c_v1.json](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_fl6v4c_v1.json) · result
- [GRADE_nt_l6v4c_v1.json](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_l6v4c_v1.json) · result
- [GRADE_nt_shed8v4d_v1.json](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_nt_shed8v4d_v1.json) · result
- [GRADE_psm_v2_v1.json](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_psm_v2_v1.json) · result
- [GRADE_psmf_v2_v1.json](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/GRADE_psmf_v2_v1.json) · result
- [fragment_governor_v1.py.txt](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_governor_v1.py.txt) · code
- [fragment_governor_v2.py.txt](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_governor_v2.py.txt) · code
- [fragment_trace_v2.py.txt](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_trace_v2.py.txt) · code
- [fragment_trace_v3.py.txt](../../wall-and-residual/residual-time-projection-switches-full-sandwich-replay-failures-12/fragment_trace_v3.py.txt) · code

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-learned-operator-products-through-tree-kernels-0-082-saving) · [Copy provenance](../../../PROVENANCE.json)

