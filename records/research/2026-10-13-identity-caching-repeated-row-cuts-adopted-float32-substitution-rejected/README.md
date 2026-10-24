# Identity caching and repeated-row cuts adopted; float32 substitution rejected because rounding exceeds savings

2026-10-13T02:22Z · decision · adopted · measured

Content repeats on the same objects are 0.062% of the metered FLOPs; the identity cache identity cache (idc1) (five sites) and the rows2 / npx1 cuts submitted byte-identically in the second and third stacks; a float64 to float32 change saved 0.026% but moved predictions by 3x that (the mean path amplifies rounding) and was rejected.

Recorded measurement:

    idc1 bill -0.0261 / -0.0247% bit for bit; s51k2 (rows2 + npx1) bill -0.1317 / -0.1347%, requests +59; f32 F1 bill -0.026% vs |d|^2 +0.07%

## Programs

- [probe_content_repeats_v1.py](probe_content_repeats_v1.py)
- [make_idc_v1.py](make_idc_v1.py)
- [../../../experiments/arithmetic-kernels/remainder-32-strided-read-variant-byte-identical-but-does/code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k2tw_x/candidate.py](../../../experiments/arithmetic-kernels/remainder-32-strided-read-variant-byte-identical-but-does/code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k2tw_x/candidate.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py](../../../instruments/native-checkpoint-port-rebuilds-submitted-estimator-byte-byte-rejects/code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py)
- [candidate.py](candidate.py)
- [candidate-faef7f17.py](candidate-faef7f17.py)
- [../../../experiments/arithmetic-kernels/merged-newborn-common-product-parked-warm-flops-0-148/code/scout_native/run_cert_v2.py](../../../experiments/arithmetic-kernels/merged-newborn-common-product-parked-warm-flops-0-148/code/scout_native/run_cert_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1766](../../TEAM_LOG_2026-10-01_to_10-22.md#L1766)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1793](../../TEAM_LOG_2026-10-01_to_10-22.md#L1793)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1824](../../TEAM_LOG_2026-10-01_to_10-22.md#L1824)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1871](../../TEAM_LOG_2026-10-01_to_10-22.md#L1871)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1885](../../TEAM_LOG_2026-10-01_to_10-22.md#L1885)

<details>
<summary>All result files and records (10)</summary>

- [cert_s136.json](cert_s136.json) · result
- [cert_s136.log](cert_s136.log) · log
- [cert_s136_guard.json](cert_s136_guard.json) · result
- [cert_tw136.json](cert_tw136.json) · result
- [cert_tw136.log](cert_tw136.log) · log
- [cert_tw136_guard.json](cert_tw136_guard.json) · result
- [cert_tw137.json](cert_tw137.json) · result
- [cert_tw137.log](cert_tw137.log) · log
- [cert_tw137_guard.json](cert_tw137_guard.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [FLOP census: two Winograd leaf products consume 67%; float64-rate work is only 0.56%](../../../experiments/flop-accounting/flop-census-two-winograd-leaf-products-consume-67-float64/README.md)
- shipped in: [Submission #335661: byte-identical repeated-row and product reductions](../../../submissions/335661/README.md)
- shipped in: [Submission #335696: reassociated native algebra and identity caching](../../../submissions/335696/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-identity-caching-repeated-row-cuts-adopted-float32-substitution-rejected) · [Copy provenance](../../../PROVENANCE.json)

