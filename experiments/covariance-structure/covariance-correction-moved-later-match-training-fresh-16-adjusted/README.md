# Covariance correction moved later to match training: fresh-16 adjusted -0.052%, out-of-sample +0.062%

2026-10-14T06:47:08Z · experiment · rejected · measured

Moving only the Torch KC add-back of the covariance to the outgoing covariance, so that within-layer birth, fourth-order and star consumers see the RC-only covariance as in training while the rounded outgoing covariance is preserved, passed the fresh pairs narrowly and failed the independent out-of-sample pairs.

Recorded measurement:

    price MAX whole log bill +0.0002849810724884816 (+0.0285%); fresh 16 pairs: log MSE -0.08152893631203333%, log bill +0.02932977346560276%, net -0.052199162846430565% (SE 0.09627047574911611%), passed; OOS 16: net +0.06200954791516716%, all 32 +0.004905192534368298%, passed=False

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/astra_kc_timing_fidelity_v1/build_native_late_kc_held_v1.py](code/lead/runs/astra_kc_timing_fidelity_v1/build_native_late_kc_held_v1.py)
- [code/lead/runs/astra_kc_timing_fidelity_v1/finite_lkc_qualification_UNVALIDATED_v1.py](code/lead/runs/astra_kc_timing_fidelity_v1/finite_lkc_qualification_UNVALIDATED_v1.py)
- [code/lead/runs/astra_kc_timing_fidelity_v1/finite_lkc_science_HELD_v5.py](code/lead/runs/astra_kc_timing_fidelity_v1/finite_lkc_science_HELD_v5.py)
- [code/lead/runs/astra_kc_timing_fidelity_v1/fragment_native_late_kc_UNVALIDATED_v1.py](code/lead/runs/astra_kc_timing_fidelity_v1/fragment_native_late_kc_UNVALIDATED_v1.py)
- [code/lead/runs/astra_kc_timing_fidelity_v1/native_lkc_qualification_judge_v1.py](code/lead/runs/astra_kc_timing_fidelity_v1/native_lkc_qualification_judge_v1.py)
- [code/lead/runs/astra_kc_timing_fidelity_v1/native_lkc_science_judge_HELD_v1.py](code/lead/runs/astra_kc_timing_fidelity_v1/native_lkc_science_judge_HELD_v1.py)
- [code/lead/runs/astra_kc_timing_fidelity_v1/native_matched_C_late_kc_OFF_UNVALIDATED_v1.py](code/lead/runs/astra_kc_timing_fidelity_v1/native_matched_C_late_kc_OFF_UNVALIDATED_v1.py)
- [code/lead/runs/astra_kc_timing_fidelity_v1/native_matched_C_late_kc_ON_UNVALIDATED_v1.py](code/lead/runs/astra_kc_timing_fidelity_v1/native_matched_C_late_kc_ON_UNVALIDATED_v1.py)
- [code/lead/runs/astra_kc_timing_fidelity_v1/owned_native_runner_v1.py](code/lead/runs/astra_kc_timing_fidelity_v1/owned_native_runner_v1.py)
- [code/release/guard_footprint_v3.py](code/release/guard_footprint_v3.py)
- [code/release/rusage.py](code/release/rusage.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [code/scout_native/certify_astra_bank_copy_v2.py](code/scout_native/certify_astra_bank_copy_v2.py)
- [code/scout_native/k4liab_s51k2tw/candidate.py](code/scout_native/k4liab_s51k2tw/candidate.py)
- [code/scout_native/run_cert_astra_bank_copy_v2.py](code/scout_native/run_cert_astra_bank_copy_v2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2682](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2682)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2706](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2706)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2708](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2708)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2734](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2734)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2735](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2735)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2777](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2777)

<details>
<summary>All result files and records (2)</summary>

- [ROOT_SCIENCE_FRESH_GRANT_v1.json](ROOT_SCIENCE_FRESH_GRANT_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-14-covariance-correction-moved-later-match-training-fresh-16-adjusted) · [Copy provenance](../../../PROVENANCE.json)

