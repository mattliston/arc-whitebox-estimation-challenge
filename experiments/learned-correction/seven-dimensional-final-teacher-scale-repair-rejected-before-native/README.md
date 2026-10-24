# Seven-dimensional final-teacher scale repair is rejected before native validation

2026-10-21T00:59:13Z · experiment · rejected · measured

Fifty-six training predictions supported a bounded scale repair, but expected teacher-error recovery was only .9248% and two of four leave-one-out cases worsened. The fit was rejected before validation or native promotion.

Recorded measurement:

    Root fixed edge6 final-teacher FD completed56training predictions at00:45:38Z; fit9b5e36ee/REL2e5d7352 was exported but REJECTED BEFORE validation/nativepromotion: all curvature ratios1.21–1.89, predictedteachererrorrecovery only.9248%, LOO2/4worse. No truth/held25training; no scalegrid followup. D/edge6_final_teacher_fd_v1 contains protocol/scripts; remote FITTED evidence copying locally. validate7/check scripts are PREPARED ONLY and MUST NOT launch this rejected fit.
    Root final-teacher edge6 FD closure sealed D/edge6_final_teacher_fd_v1/REJECTED_v1.json at01:10:29Z; all56predictions/28guards and rejected sources copied in A/cloud_linux_markline_v1/root_edge6_fd_receipts_v1 (FINAL_COPY227b2440). Never launch its rejected scale sources/check/validation scripts. Current old Macqueue77740 still active; three original judges seen01:20Z. Root owns per-layer edge checks; V shared/head repairs; Bill live graph/expensive-old-accuracy audit; joint VM dispatch/health/freshtruth. Continue all-out research; no submission authorized for present small bank.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/edge6_final_teacher_fd_v1/launch_v2.py](code/edge6_final_teacher_fd_v1/launch_v2.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l12_plus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l12_plus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l13_minus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l13_minus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l14_plus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l14_plus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l12_minus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l12_minus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/validate7_v1.py](code/edge6_final_teacher_fd_v1/validate7_v1.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l13_plus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l13_plus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/check_fitted_scale_v1.py](code/edge6_final_teacher_fd_v1/check_fitted_scale_v1.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l8_plus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l8_plus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l9_minus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l9_minus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l11_plus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l11_plus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l11_minus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l11_minus_v1_tail.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l14_minus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l14_minus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/prepare_v1.py](code/edge6_final_teacher_fd_v1/prepare_v1.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l10_plus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l10_plus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/fit_v1.py](code/edge6_final_teacher_fd_v1/fit_v1.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l8_minus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l8_minus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/launch_v1.py](code/edge6_final_teacher_fd_v1/launch_v1.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l9_plus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l9_plus_v1_tail.py)
- [code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l10_minus_v1_tail.py](code/edge6_final_teacher_fd_v1/edge6_teacher_fd_l10_minus_v1_tail.py)
- [code/edge_hidden6_nonlinear_v1/native_hidden6_v1/candidate_msg8_edge_hidden6_nonlinear_science_v1.py](code/edge_hidden6_nonlinear_v1/native_hidden6_v1/candidate_msg8_edge_hidden6_nonlinear_science_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6427](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6427)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6442](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6442)

<details>
<summary>All result files and records (25)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [REJECTED_v1.json](REJECTED_v1.json) · result
- [BUILD_v1.json](BUILD_v1.json) · result
- [PLAN_v1.json](PLAN_v1.json) · result
- [edge6_teacher_fd_l8_plus_v1_LAUNCH.json](edge6_teacher_fd_l8_plus_v1_LAUNCH.json) · result
- [edge6_teacher_fd_l9_plus_v1_LAUNCH.json](edge6_teacher_fd_l9_plus_v1_LAUNCH.json) · result
- [edge6_teacher_fd_l11_minus_v1_LAUNCH.json](edge6_teacher_fd_l11_minus_v1_LAUNCH.json) · result
- [FIT_WAITER_LAUNCH_v1.json](FIT_WAITER_LAUNCH_v1.json) · result
- [COPY_20261021T004540Z.json](COPY_20261021T004540Z.json) · result
- [edge6_teacher_fd_l14_minus_v1_LAUNCH.json](edge6_teacher_fd_l14_minus_v1_LAUNCH.json) · result
- [FINAL_COPY_v1.json](FINAL_COPY_v1.json) · result
- [edge6_teacher_fd_l11_plus_v1_LAUNCH.json](edge6_teacher_fd_l11_plus_v1_LAUNCH.json) · result
- [edge6_teacher_fd_l10_plus_v1_LAUNCH.json](edge6_teacher_fd_l10_plus_v1_LAUNCH.json) · result
- [edge6_teacher_fd_l13_plus_v1_LAUNCH.json](edge6_teacher_fd_l13_plus_v1_LAUNCH.json) · result
- [edge6_teacher_fd_l12_plus_v1_LAUNCH.json](edge6_teacher_fd_l12_plus_v1_LAUNCH.json) · result
- [edge6_teacher_fd_l14_plus_v1_LAUNCH.json](edge6_teacher_fd_l14_plus_v1_LAUNCH.json) · result
- [edge6_teacher_fd_l10_minus_v1_LAUNCH.json](edge6_teacher_fd_l10_minus_v1_LAUNCH.json) · result
- [edge6_teacher_fd_l13_minus_v1_LAUNCH.json](edge6_teacher_fd_l13_minus_v1_LAUNCH.json) · result
- [LAUNCH_v1.json](LAUNCH_v1.json) · result
- [edge6_teacher_fd_l8_minus_v1_LAUNCH.json](edge6_teacher_fd_l8_minus_v1_LAUNCH.json) · result
- [BASELINE_REUSE_PROOF_v1.json](BASELINE_REUSE_PROOF_v1.json) · result
- [edge6_teacher_fd_l12_minus_v1_LAUNCH.json](edge6_teacher_fd_l12_minus_v1_LAUNCH.json) · result
- [FITTED_v1.json](FITTED_v1.json) · result
- [NUMPY_FREE_IMPORT_v1.json](NUMPY_FREE_IMPORT_v1.json) · result
- [edge6_teacher_fd_l9_minus_v1_LAUNCH.json](edge6_teacher_fd_l9_minus_v1_LAUNCH.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-seven-dimensional-final-teacher-scale-repair-rejected-before-native) · [Copy provenance](../../../PROVENANCE.json)

