# Norm-fourth proxy conservation and fourth-cumulant transport: adjusted +0.0002%/+0.0006%

2026-10-14T08:18:50Z · experiment · rejected · measured

Two fixed scalar fourth-order constraints on the radial carry at layers 1 to 14 were tried: Q4 conserves a raw non-central norm-fourth closure proxy across the learned head's mean and covariance update, and CENTRAL transports the central fourth-cumulant normalisation from the original pair numerator to the actual outgoing covariance. Both priced at under 0.002% bill and changed nothing measurable.

Recorded measurement:

    price MAX log bill: Q4 0.0014801962099865763, CENTRAL 0.00022770301776336594; fresh 16: Q4 mean log net +0.00022263204580544237% (SE 0.0006827437903173875%), CENTRAL +0.000580719316401717% (SE 0.0007436887764583048%); passed=false for both

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_raw_norm_fourth_v1/fragment_raw_q4_UNVALIDATED_v3.py](code/lead/runs/astra_raw_norm_fourth_v1/fragment_raw_q4_UNVALIDATED_v3.py)
- [code/lead/runs/astra_raw_norm_fourth_v1/finite_q4_qualification_UNVALIDATED_v1.py](code/lead/runs/astra_raw_norm_fourth_v1/finite_q4_qualification_UNVALIDATED_v1.py)
- [code/lead/runs/astra_radial_shared_science_v1/finite_radial_science_HELD_v6.py](code/lead/runs/astra_radial_shared_science_v1/finite_radial_science_HELD_v6.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_radial_shared_science_v1/native_radial_science_judge_HELD_v1.py](code/lead/runs/astra_radial_shared_science_v1/native_radial_science_judge_HELD_v1.py)
- [code/lead/runs/astra_raw_norm_fourth_v1/native_q4_qualification_judge_v1.py](code/lead/runs/astra_raw_norm_fourth_v1/native_q4_qualification_judge_v1.py)
- [code/lead/runs/astra_raw_norm_fourth_v1/owned_native_runner_v1.py](code/lead/runs/astra_raw_norm_fourth_v1/owned_native_runner_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_card/central_fourth_normalization_held_astra_v1/finite_cfn_qualification_UNVALIDATED_v3.py](code/scout_card/central_fourth_normalization_held_astra_v1/finite_cfn_qualification_UNVALIDATED_v3.py)
- [code/scout_card/central_fourth_normalization_held_astra_v1/native_cfn_qualification_judge_v1.py](code/scout_card/central_fourth_normalization_held_astra_v1/native_cfn_qualification_judge_v1.py)
- [code/scout_card/central_fourth_normalization_held_astra_v1/owned_native_runner_v1.py](code/scout_card/central_fourth_normalization_held_astra_v1/owned_native_runner_v1.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/k4liab_s51k2tw/candidate.py](code/scout_native/k4liab_s51k2tw/candidate.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2727](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2727)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2742](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2742)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2744](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2744)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2779](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2779)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2826](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2826)

<details>
<summary>All result files and records (2)</summary>

- [SOURCE_TINY_FIXED6_PEER_v1.json](SOURCE_TINY_FIXED6_PEER_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-norm-fourth-proxy-conservation-fourth-cumulant-transport-adjusted-0) · [Copy provenance](../../../PROVENANCE.json)

