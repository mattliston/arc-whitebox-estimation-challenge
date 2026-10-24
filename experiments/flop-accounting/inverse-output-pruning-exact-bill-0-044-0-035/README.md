# Inverse output pruning: exact, bill -0.044%/-0.035% with 1,979/2,312 additional requests

2026-10-14T04:47:28Z · experiment · rejected · derived

Inverse output pruning (IOP) omits the elementwise inverse operations that feed only padded output rows and columns, keeping exact operation order on every retained scalar. The six-call qualification was byte-exact on all sixteen layers with a measured bill credit, but it raised the charged request count by about two thousand per network; the fired #336075 stack carried it disabled.

Recorded measurement:

    whole bills lower by 113,432,739 / 93,994,698 FLOPs (0.0442004% / 0.0350034%) on 7100136/7100137; geometric credit 0.03960199904643469%, worst 0.03500337534732001%; charged requests +1,979 / +2,312

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_inverse_output_prune_v1/fragment_inverse_output_prune_UNVALIDATED_v1.py](code/lead/runs/astra_inverse_output_prune_v1/fragment_inverse_output_prune_UNVALIDATED_v1.py)
- [code/lead/runs/astra_inverse_output_prune_v1/finite_iop_qualification_v1.py](code/lead/runs/astra_inverse_output_prune_v1/finite_iop_qualification_v1.py)
- [code/lead/runs/astra_inverse_output_prune_v1/build_source_v1.py](code/lead/runs/astra_inverse_output_prune_v1/build_source_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py](code/engineer/a62_modules/s49/s56ubq1tw_x/candidate.py)
- [code/lead/runs/astra_inverse_output_prune_v1/inverse_plan_HELD_v2.py](code/lead/runs/astra_inverse_output_prune_v1/inverse_plan_HELD_v2.py)
- [code/lead/runs/astra_inverse_output_prune_v1/native_iop_qualification_judge_v1.py](code/lead/runs/astra_inverse_output_prune_v1/native_iop_qualification_judge_v1.py)
- [code/lead/runs/astra_inverse_output_prune_v1/owned_native_runner_v1.py](code/lead/runs/astra_inverse_output_prune_v1/owned_native_runner_v1.py)
- [code/lead/runs/astra_inverse_output_prune_v1/tiny_fragment_dirty_v1.py](code/lead/runs/astra_inverse_output_prune_v1/tiny_fragment_dirty_v1.py)
- [code/lead/runs/astra_inverse_output_prune_v1/tiny_fragment_dirty_v2.py](code/lead/runs/astra_inverse_output_prune_v1/tiny_fragment_dirty_v2.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2635](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2635)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2658](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2658)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2660](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2660)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Joint Gaussian T22 reuse plus inverse output pruning: exact, bill -0.069%/-0.059% (mean -0.064%), requests +2,119/+2,452](../joint-gaussian-t22-reuse-plus-inverse-output-pruning-exact/README.md)
- ← rejects: [Stack with inverse output pruning fails the request gate (loss risk 88.33%, 57,693/52,643 requests); pruning dropped](../../../records/research/2026-10-14-stack-inverse-output-pruning-fails-request-gate-loss-risk/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-inverse-output-pruning-exact-bill-0-044-0-035) · [Copy provenance](../../../PROVENANCE.json)

