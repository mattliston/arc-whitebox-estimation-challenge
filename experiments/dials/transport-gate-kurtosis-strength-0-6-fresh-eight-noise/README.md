# Transport-gate kurtosis strength 0.6: fresh-eight noise-free MSE -0.072% at identical FLOPs

2026-10-16T01:19:51Z · experiment · adopted · measured

The fixed reduction from 1.0 to 0.6 passed the nominal negative-mean rule on full rank 384, but uncertainty was larger than the gain. Other head, edge, birth, output, and late-capacity dials failed the screening.

Recorded measurement:

    U-log-.072%,SE.246,3/8lower,range-.94..+.88;plain-.064%;bill0.000%;birthscale1.0+1.3%,1.069+4.1%;output.5/1.0+.31/+.33%;L15fullu-.005%at+.56%bill

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_acc_dials_v1/grade_x24_v1.py](code/lead/runs/opus_acc_dials_v1/grade_x24_v1.py)
- [code/lead/runs/opus_acc_dials_v1/candidate_full384_gk4_060_v1.py](code/lead/runs/opus_acc_dials_v1/candidate_full384_gk4_060_v1.py)
- [code/lead/runs/opus_acc_dials_v1/analyze_v1.py](code/lead/runs/opus_acc_dials_v1/analyze_v1.py)
- [code/lead/runs/opus_acc_dials_v1/analyze_v2.py](code/lead/runs/opus_acc_dials_v1/analyze_v2.py)
- [code/lead/runs/opus_acc_dials_v1/dial_judge_v1.py](code/lead/runs/opus_acc_dials_v1/dial_judge_v1.py)
- [code/lead/runs/opus_acc_dials_v1/dial_judge_v2.py](code/lead/runs/opus_acc_dials_v1/dial_judge_v2.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3629](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3629)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3627](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3627)

<details>
<summary>All result files and records (7)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [X24_DECISION_v1.json](X24_DECISION_v1.json) · result
- [analysis_latest.json](analysis_latest.json) · result
- [jobs_b1.json](jobs_b1.json) · result
- [jobs_b2.json](jobs_b2.json) · result
- [jobs_b3.json](jobs_b3.json) · result
- [jobs_b4.json](jobs_b4.json) · result

</details>

## Related work

- shipped in: [Submission #336367: layer-10 rank 384, compact kernels and exact request cuts; seven runtime failures](../../../submissions/336367/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-transport-gate-kurtosis-strength-0-6-fresh-eight-noise) · [Copy provenance](../../../PROVENANCE.json)

