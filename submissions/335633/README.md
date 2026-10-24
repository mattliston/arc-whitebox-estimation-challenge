# Submission #335633: corrected transport gate plus byte-identical transform caching

2026-10-12T22:34:48Z · submission · adopted · measured

The first stacked twin: the skew- and kurtosis-corrected transport gate (gk) transport gate (moment-closure estimator delta) and the float32 zero-fill and Winograd-transform caching package (SN3) metered FLOPs package (native delta) on candidate generation 51 (S51)'s files, sealed on 8 fresh networks and fast-gated; the board confirmed the sum (final MSE -0.198%, metered FLOPs -0.152%).

Recorded measurement:

    1.53890e-9 (-0.357% vs #335621; paired mean log -0.359%, se 0.098, 33/50); final MSE 1.3395e-8; util 0.11489; 0 failed; residual max 0.342 s; seal -0.284% (se 0.184, 6 of 8); official public score 1.5388232379587904e-09; official utilisation 0.11488500111536268; failed networks 0

Public score: 1.5388232379587904e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [candidate.py](candidate.py)
- [candidate-fb4456a7.py](candidate-fb4456a7.py)
- [run_first_stack_v3.sh](run_first_stack_v3.sh)
- [../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/engineer/a41_base/fragment_gsk_v1.py](../../instruments/estimator-builder-composes-implementation-fragments-checks-tensor-contractions/code/engineer/a41_base/fragment_gsk_v1.py)
- [../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_gsk_v1.py](../../instruments/generated-refit-after-drop-rounds-connect-sizing-porting-sealed/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/make_gsk_v1.py)
- [../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/parity_verdict_v3.py](../../instruments/native-route-table-updates-automatically-certified-parity-verdicts/code/opus_reexam_20261001_v1/engineer/a62_modules/parity_verdict_v3.py)
- [candidate-5a655257.py](candidate-5a655257.py)
- [candidate-8eae0a4b.py](candidate-8eae0a4b.py)
- [candidate-326f9261.py](candidate-326f9261.py)
- [../335508/candidate-fb277a8a.py](../335508/candidate-fb277a8a.py)

<details>
<summary>Additional programs and dependencies (7)</summary>

- [../../records/research/2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/candidate.py](../../records/research/2026-10-12-combined-candidate-workflow-applies-verified-changes-seals-fresh-networks/candidate.py)
- [../../records/research/2026-10-12-refit-parity-correlation-0-893-size-1-029-research/candidate-f314d8ea.py](../../records/research/2026-10-12-refit-parity-correlation-0-893-size-1-029-research/candidate-f314d8ea.py)
- [../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/release/guard_footprint_v2.py](../../instruments/native-per-layer-routes-added-kink-plane-scale-young/code/opus_reexam_20261001_v1/release/guard_footprint_v2.py)
- [../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py](../../instruments/release-qualification-checks-packaging-boundary-instructions-setup-timing-worker/code/rusage.py)
- [../../records/research/2026-10-12-float32-zero-fills-cached-winograd-transforms-adopted-about-0/fragment_sn3_v1.py](../../records/research/2026-10-12-float32-zero-fills-cached-winograd-transforms-adopted-about-0/fragment_sn3_v1.py)
- [../../experiments/arithmetic-kernels/merged-newborn-common-product-parked-warm-flops-0-148/code/scout_native/run_cert_v2.py](../../experiments/arithmetic-kernels/merged-newborn-common-product-parked-warm-flops-0-148/code/scout_native/run_cert_v2.py)
- [candidate-fa827fd2.py](candidate-fa827fd2.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1660](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1660)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1684](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1684)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1688](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1688)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1716](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1716)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1739](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1739)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1740](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1740)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1777](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1777)
- [SUBMISSION_LEDGER.md#L953](../../records/SUBMISSION_LEDGER.md#L953)
- [SUBMISSION_LEDGER.md#L954](../../records/SUBMISSION_LEDGER.md#L954)
- [SUBMISSION_LEDGER.md#L955](../../records/SUBMISSION_LEDGER.md#L955)
- [SUBMISSION_LEDGER.md#L956](../../records/SUBMISSION_LEDGER.md#L956)

<details>
<summary>All result files and records (6)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [stage_a.json](stage_a.json) · record
- [receipt.json](receipt.json) · record
- [eval_335633.json](eval_335633.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Float32 zero fills and cached Winograd transforms adopted: about -0.24% FLOPs, byte-identical](../../records/research/2026-10-12-float32-zero-fills-cached-winograd-transforms-adopted-about-0/README.md)
- ← shipped in: [Skew- and kurtosis-corrected transport gate adopted: -0.348% pooled MSE on 24 fresh networks](../../records/research/2026-10-12-skew-kurtosis-corrected-transport-gate-adopted-0-348-pooled/README.md)
- ← supersedes: [Submission #335621: higher-request form of the same refit, qualified through the fast release gate](../335621/README.md)
- supersedes: [Submission #335661: byte-identical repeated-row and product reductions](../335661/README.md)

[Research graph](../../graph/index.html#node=submission-335633) · [Copy provenance](../../PROVENANCE.json)

