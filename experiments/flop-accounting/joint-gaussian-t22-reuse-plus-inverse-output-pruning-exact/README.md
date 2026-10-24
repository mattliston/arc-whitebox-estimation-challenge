# Joint Gaussian T22 reuse plus inverse output pruning: exact, bill -0.069%/-0.059% (mean -0.064%), requests +2,119/+2,452

2026-10-14T05:29:11Z · experiment · inconclusive · derived

The two exact fragments were composed and qualified together on the same two networks with observed counter sums rather than added percentages. Adding the dense covariance reuse gave a combined qualified compute saving of 0.1066% against the current file.

Recorded measurement:

    bill reductions 0.069405311% / 0.059091537%, geometric mean 0.06424855686245623% (worst 0.05909153669312328%); credits 178,116,771 / 158,678,730 FLOPs; charged requests +2,119 / +2,452; with dense reuse: dense incremental 0.042397493%, combined GTR+IOP+dense 0.106618810% geometric on 7100136/7100137

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_gtr_iop_joint_v1/finite_joint_qualification_HELD_v1.py](code/lead/runs/astra_gtr_iop_joint_v1/finite_joint_qualification_HELD_v1.py)
- [code/lead/runs/astra_dense_gtr_iop_qualification_v1/finite_dense_qualification_HELD_v1.py](code/lead/runs/astra_dense_gtr_iop_qualification_v1/finite_dense_qualification_HELD_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py](code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py)
- [code/lead/runs/astra_dense_gtr_iop_qualification_v1/native_dense_qualification_judge_HELD_v1.py](code/lead/runs/astra_dense_gtr_iop_qualification_v1/native_dense_qualification_judge_HELD_v1.py)
- [code/lead/runs/astra_dense_gtr_iop_qualification_v1/owned_native_runner_v1.py](code/lead/runs/astra_dense_gtr_iop_qualification_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_gtr_iop_joint_v1/candidate_joint_ON_v1.py](code/lead/runs/astra_gtr_iop_joint_v1/candidate_joint_ON_v1.py)
- [code/lead/runs/astra_gtr_iop_joint_v1/native_joint_qualification_judge_v1.py](code/lead/runs/astra_gtr_iop_joint_v1/native_joint_qualification_judge_v1.py)
- [code/lead/runs/astra_gtr_iop_joint_v1/owned_native_runner_v1.py](code/lead/runs/astra_gtr_iop_joint_v1/owned_native_runner_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2686](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2686)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2713](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2713)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2715](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2715)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2828](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2828)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2832](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2832)

<details>
<summary>All result files and records (3)</summary>

- [IMPORT_AND_COMPOSITION_TINY_v1.json](IMPORT_AND_COMPOSITION_TINY_v1.json) · record
- [COMPOSITION_SOURCE_MANIFEST_HELD_v1.json](COMPOSITION_SOURCE_MANIFEST_HELD_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Gaussian second-order common-product reuse: exact, 64,684,032 FLOPs saved for 140 additional requests](../gaussian-second-order-common-product-reuse-exact-64-684/README.md)
- ← enables: [Inverse output pruning: exact, bill -0.044%/-0.035% with 1,979/2,312 additional requests](../inverse-output-pruning-exact-bill-0-044-0-035/README.md)
- shipped in: [Submission #336075: feasible head mean, dense covariance reuse and Gaussian T22 reuse on #335920, 1.51317e-9](../../../submissions/336075/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-joint-gaussian-t22-reuse-plus-inverse-output-pruning-exact) · [Copy provenance](../../../PROVENANCE.json)

