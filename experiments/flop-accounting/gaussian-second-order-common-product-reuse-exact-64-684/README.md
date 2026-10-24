# Gaussian second-order common-product reuse: exact, 64,684,032 FLOPs saved for 140 additional requests

2026-10-14T02:36:37Z · experiment · adopted · derived

The Gaussian T22 reuse (GTR) reuses a common product of the Gaussian transform instead of recomputing it over 14 layers. Its six-call native qualification (parent, off and on on two fixed networks) reproduced all sixteen layer predictions byte-exactly and measured the projected credit exactly, at the cost of 140 extra charged requests per network. It entered the #336075 stack.

Recorded measurement:

    whole-bill credit 64,684,032 FLOPs per network (2Q+16H counter over 14 layers), +140 charged requests; bills -0.0252048997% and -0.0240881613% on networks 7100136/7100137; geometric bill reduction 0.024646532096096595%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_gaussian_t22_reuse_v1/fragment_gaussian_t22_reuse_UNVALIDATED_v3.py](code/lead/runs/astra_gaussian_t22_reuse_v1/fragment_gaussian_t22_reuse_UNVALIDATED_v3.py)
- [code/lead/runs/astra_gaussian_t22_reuse_v1/finite_gtr_qualification_v1.py](code/lead/runs/astra_gaussian_t22_reuse_v1/finite_gtr_qualification_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py](code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py)
- [code/lead/runs/astra_gaussian_t22_reuse_v1/native_gtr_qualification_judge_v1.py](code/lead/runs/astra_gaussian_t22_reuse_v1/native_gtr_qualification_judge_v1.py)
- [code/lead/runs/astra_gaussian_t22_reuse_v1/owned_native_runner_v1.py](code/lead/runs/astra_gaussian_t22_reuse_v1/owned_native_runner_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2456](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2456)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2532](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2532)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2536](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2536)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2548](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2548)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2550](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2550)

<details>
<summary>All result files and records (3)</summary>

- [SOURCE_READY_v1.json](SOURCE_READY_v1.json) · record
- [QUALIFICATION_OWNER_ACCEPTANCE_PREP_v1.json](QUALIFICATION_OWNER_ACCEPTANCE_PREP_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Joint Gaussian T22 reuse plus inverse output pruning: exact, bill -0.069%/-0.059% (mean -0.064%), requests +2,119/+2,452](../joint-gaussian-t22-reuse-plus-inverse-output-pruning-exact/README.md)
- shipped in: [Submission #336075: feasible head mean, dense covariance reuse and Gaussian T22 reuse on #335920, 1.51317e-9](../../../submissions/336075/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-gaussian-second-order-common-product-reuse-exact-64-684) · [Copy provenance](../../../PROVENANCE.json)

