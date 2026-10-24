# Trailing 64 read coordinates cut only at layers 13-14: 24-network adjusted +0.376%

2026-10-16T04:37:31Z · experiment · rejected · measured

The later-only cut on greedy bases lost real signal and failed both pooled and final-eight rules, despite leaving layers 0-12 byte-identical.

Recorded measurement:

    xMSE+.576%,24/24positive;bill-.200%;net+.376%,SE.076,3/24lower;last8+.229%,SE.063;requests+136first/+198warm

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_u64late_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/grade_u64late_v1.py](code/grade_u64late_v1.py)
- [code/cand_2b108_gb128r_u64late_v1.py](code/cand_2b108_gb128r_u64late_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3869](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3869)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [U64LATE_RESULT_v1.json](U64LATE_RESULT_v1.json) · result
- [fragment_u64late_L13_14_v1.py.txt](fragment_u64late_L13_14_v1.py.txt) · result
- [PREREG_U64LATE_v1.json](PREREG_U64LATE_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-trailing-64-read-coordinates-cut-only-layers-13-14) · [Copy provenance](../../../PROVENANCE.json)

