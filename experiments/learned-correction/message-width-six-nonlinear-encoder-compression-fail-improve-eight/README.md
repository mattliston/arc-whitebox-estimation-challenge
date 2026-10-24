# Message width six and nonlinear encoder compression fail to improve the eight-channel parent

2026-10-20T23:37:21Z · experiment · rejected · measured

Full-32 tests found message width six trailed width eight and nonlinear width-64 encoder compression was worse. Full-head repair on actual compressed states also failed, while edge hidden widths 12/10/8 did not improve the earlier edge parent.

Recorded measurement:

    Closedfull32Linux: msg6-.68277551%held-.69145923 trailsmsg8; nonlinear64+.55047115%held+.59034558 reject. Originalenc96/msg8joint-.228675%; actualstudentfull-headrepair8f82065d whole32-.128717%held-.101481%raw+.537996%bill-.663176% worse; fullguardPASS notharnessfailure. Billauthorizedterminal-onlyL15 headrepair (trainingLOO−13.6%, all-layerLOO+4.03%) asseparatefixedliteralcandidate; noheldtruthfit. Edgeh12-.11833868%held-.09416315;h10-.12596915%held-.16853292;h8+.04061219%held+.01518104 allworsethanEH14. Noadditionaledgebankpiece.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_vm_frontier_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cloud_return_v1/export_nonlinear96_native_v1/candidate_enc96_msg8_nonlinear_teacher_sci_v1.py](code/cloud_return_v1/export_nonlinear96_native_v1/candidate_enc96_msg8_nonlinear_teacher_sci_v1.py)
- [code/cloud_return_v1/export_nonlinear_native_v1/candidate_enc64_msg8_nonlinear_teacher_sci_v1.py](code/cloud_return_v1/export_nonlinear_native_v1/candidate_enc64_msg8_nonlinear_teacher_sci_v1.py)
- [code/cloud_return_v1/export_student_nonlinear_native_v1/candidate_enc96_msg8_student_nonlinear_teacher_sci_v1.py](code/cloud_return_v1/export_student_nonlinear_native_v1/candidate_enc96_msg8_student_nonlinear_teacher_sci_v1.py)
- [code/cloud_return_v1/fit_msg6_native_v1/candidate_msg6_teacher_sci_v1.py](code/cloud_return_v1/fit_msg6_native_v1/candidate_msg6_teacher_sci_v1.py)
- [code/export_frontier_nonlinear_v1.py](code/export_frontier_nonlinear_v1.py)
- [code/export_frontier_nonlinear_v2.py](code/export_frontier_nonlinear_v2.py)
- [code/export_frontier_student_nonlinear_v1.py](code/export_frontier_student_nonlinear_v1.py)
- [code/fit_frontier_head_repair_v1.py](code/fit_frontier_head_repair_v1.py)
- [code/fit_frontier_message6_v1.py](code/fit_frontier_message6_v1.py)
- [code/fit_frontier_nonlinear_v1.py](code/fit_frontier_nonlinear_v1.py)
- [code/fit_frontier_nonlinear_v2.py](code/fit_frontier_nonlinear_v2.py)
- [code/fit_frontier_student_nonlinear_v1.py](code/fit_frontier_student_nonlinear_v1.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/fit_frontier_v1.py](code/fit_frontier_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6385](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6385)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-message-width-six-nonlinear-encoder-compression-fail-improve-eight) · [Copy provenance](../../../PROVENANCE.json)

