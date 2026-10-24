# Correction-only learned factorization: import failure followed by displacement RMS 0.0320, rejected

2026-10-14T01:46:38Z · experiment · rejected · measured

A correction-only version of the E3 energy factor of the learned correction was qualified natively in six calls. The first attempt stopped on an identity assertion at import; after a source alias repair the on-arm exceeded the 0.001 all-layer displacement cap although its bill counter matched.

Recorded measurement:

    first attempt: assertion failed at OFF import; second: displacement cap 0.001 failed (max 0.03198249, final 0.02409658), exact bill counter 552,394,740

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_e31_e3_factor_v1/fragment_e31_e3_UNVALIDATED_v4.py](code/lead/runs/astra_e31_e3_factor_v1/fragment_e31_e3_UNVALIDATED_v4.py)
- [code/lead/runs/astra_e31_e3_factor_v1/finite_e31_qualification_v2.py](code/lead/runs/astra_e31_e3_factor_v1/finite_e31_qualification_v2.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/s56k1tw_x/candidate.py](code/engineer/a62_modules/s49/s56k1tw_x/candidate.py)
- [code/lead/runs/astra_e31_e3_factor_v1/native_e31_qualification_judge_v1.py](code/lead/runs/astra_e31_e3_factor_v1/native_e31_qualification_judge_v1.py)
- [code/lead/runs/astra_e31_e3_factor_v1/owned_native_runner_v1.py](code/lead/runs/astra_e31_e3_factor_v1/owned_native_runner_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)
- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2462](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2462)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2475](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2475)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2487](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2487)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2498](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2498)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2500](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2500)

<details>
<summary>All result files and records (2)</summary>

- [QUALIFICATION_CAP_FAIL_RETIREMENT_v1.json](QUALIFICATION_CAP_FAIL_RETIREMENT_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-correction-only-learned-factorization-import-failure-followed-displacement-rms) · [Copy provenance](../../../PROVENANCE.json)

