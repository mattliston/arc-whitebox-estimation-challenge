# Repeated public and generated gradient refits converge to the existing point

2026-10-11T11:01Z · lesson · rejected · measured

public-gradient refit after Nd (Ne) (public gradients at 32-network ridge-re-sized cut-simulated refit (Nd)) froze to Nd within rounding with every leave-one-out >= 0; generated-gradient refit after Nd (Nf) (gradients from 64 generated networks the learned parts never trained on) did the same: the population gradient at Nd is small in every family, so the refit engine is converged on today's inputs.

Recorded measurement:

    Ne: all LOO >= 0, h17 alone +2.46%; Nf: head arrays -0.004% (se 0.011), edge scalars -0.005%, eold+qb +0.136%; generated t^2 0.6-6.6

## Programs

- [jmix_generated_v6.py](jmix_generated_v6.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_manifest_v1.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_generated_v1.py)
- [../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py](../../../instruments/variable-sample-truth-consumer-validates-sample-counts-sealed-network/code/opus_reexam_20261001_v1/trainer/otr_truth_records_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1049](../../TEAM_LOG_2026-10-01_to_10-22.md#L1049)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1074](../../TEAM_LOG_2026-10-01_to_10-22.md#L1074)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1075](../../TEAM_LOG_2026-10-01_to_10-22.md#L1075)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1076](../../TEAM_LOG_2026-10-01_to_10-22.md#L1076)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1077](../../TEAM_LOG_2026-10-01_to_10-22.md#L1077)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1078](../../TEAM_LOG_2026-10-01_to_10-22.md#L1078)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1079](../../TEAM_LOG_2026-10-01_to_10-22.md#L1079)

<details>
<summary>All result files and records (4)</summary>

- [FROZEN_POINT_v1.json](FROZEN_POINT_v1.json) · record
- [FROZEN_POINT_v1-3ed8ba83.json](FROZEN_POINT_v1-3ed8ba83.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [band_table_v1.out](band_table_v1.out) · result

</details>

## Related work

- enables: [Birth pruning, low-rank young reads, covariance-channel corrections and edge inputs fail measured gates](../../../experiments/third-order-planes/birth-pruning-low-rank-young-reads-covariance-channel-corrections/README.md)
- enables: [First refit-after-drop strength round adopted: -0.223% on fresh networks at zero added FLOPs](../2026-10-11-first-refit-after-drop-strength-round-adopted-0-223/README.md)
- ← enables: [Ridge re-sizing on 32 networks adopted: pooled independent 64-network MSE change -0.323%](../2026-10-11-ridge-re-sizing-32-networks-adopted-pooled-independent-64/README.md)

[Research graph](../../../graph/index.html#node=2026-10-11-repeated-public-generated-gradient-refits-converge-existing-point) · [Copy provenance](../../../PROVENANCE.json)

