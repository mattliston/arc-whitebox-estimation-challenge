# Partially sized refit superseded so the next round absorbs the corrected transport gate

2026-10-12T22:47:45Z · decision · adopted · measured

candidate generation 52 (S52) (arms arrp + h17 only, dials none, no skew- and kurtosis-corrected transport gate (gk)) had about 170 of 384 sizing evals left at 2 workers with a verdict due 01:30-02:30Z, and candidate generation 53 (S53) would have waited for it; the lead superseded it: the chain was stopped at 22:50:50Z and STOP written, 221 of 384 sizing results, the 64 gradients and the arms archived (no freeze, no verdict, so no stage 2, port or release; the port loop skipped it), WORKER_CAP went 2 -> 4, and S53 started from candidate generation 51 (S51)'s frozen point with gk11 absorbed, rule (c) once and the routes as in ROUTES.json; a4k31 stayed out of S53.

Recorded measurement:

    S52: 221 of 384 sizing results archived, 64 gradients, arms arrp / h17; S53 start = S51 frozen point, ABSORB.json f7fed87e2afd, WORKER_CAP 2 -> 4; expected S52 v2 -0.03..-0.06%

## Programs

- [chain_s53_v20.sh](chain_s53_v20.sh)
- [make_s53_v11.py](make_s53_v11.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../2026-10-12-refit-round-rules-remove-re-sized-arms-second-stage/aggregate_spec_v1.py](../2026-10-12-refit-round-rules-remove-re-sized-arms-second-stage/aggregate_spec_v1.py)
- [../2026-10-12-refit-round-rules-remove-re-sized-arms-second-stage/ptload.py](../2026-10-12-refit-round-rules-remove-re-sized-arms-second-stage/ptload.py)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/trainer/cert/RAD/S53/make_s53_v3.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/trainer/cert/RAD/S53/make_s53_v3.py)
- [absorb_v1.py](absorb_v1.py)
- [../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/make_exportable.py](../../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer/cert/RAD/make_exportable.py)
- [../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py](../../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/trainer/cert/RAD/rad_v6.py)
- [../2026-10-12-zero-step-refit-arm-receives-weight-16-54-rounding/zero_arms_v1.py](../2026-10-12-zero-step-refit-arm-receives-weight-16-54-rounding/zero_arms_v1.py)
- [../2026-10-11-first-refit-after-drop-strength-round-adopted-0-223/jmix_generated_v7.py](../2026-10-11-first-refit-after-drop-strength-round-adopted-0-223/jmix_generated_v7.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [../../../instruments/native-parameter-export-supports-ten-fitted-dials-gradient-direction/code/opus_reexam_20261001_v1/trainer/export.py](../../../instruments/native-parameter-export-supports-ten-fitted-dials-gradient-direction/code/opus_reexam_20261001_v1/trainer/export.py)
- [export_dials_s46_v8.py](export_dials_s46_v8.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1674](../../TEAM_LOG_2026-10-01_to_10-22.md#L1674)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1675](../../TEAM_LOG_2026-10-01_to_10-22.md#L1675)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1746](../../TEAM_LOG_2026-10-01_to_10-22.md#L1746)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1753](../../TEAM_LOG_2026-10-01_to_10-22.md#L1753)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1757](../../TEAM_LOG_2026-10-01_to_10-22.md#L1757)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1762](../../TEAM_LOG_2026-10-01_to_10-22.md#L1762)

<details>
<summary>All result files and records (3)</summary>

- [SUPERSEDED_v1.json](../../../experiments/learned-correction/re-sizing-previously-fitted-arms-gives-no-fresh-network/SUPERSEDED_v1.json) · result
- [STOP](STOP) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-partially-sized-refit-superseded-so-next-round-absorbs-corrected) · [Copy provenance](../../../PROVENANCE.json)

