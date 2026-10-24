# Cross-half truth product removes Monte Carlo noise that masks about one quarter of local MSE changes

2026-10-16T00:54:51Z · lesson · adopted · measured

The unbiased error estimate U=mean((prediction-A)*(prediction-B)) from independent truth halves matched public error levels. Subsequent accuracy decisions used this noise-corrected objective.

Recorded measurement:

    plain1.786e-8,cross-half1.340e-8,public1.336e-8;16Mtruths~25%MSE;plainchanges~x0.75

## Programs

- [../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/birth_v1.py](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/birth_v1.py)
- [../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/birth_xhalf_v1.py](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/birth_xhalf_v1.py)
- [../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/born_tbins_v1.py](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/born_tbins_v1.py)
- [../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/features_v1.py](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/features_v1.py)
- [../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/harvest_v1.py](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/harvest_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3613](../../TEAM_LOG_2026-10-01_to_10-22.md#L3613)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3608](../../TEAM_LOG_2026-10-01_to_10-22.md#L3608)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [BIRTH_v1.json](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/BIRTH_v1.json) · result
- [BORN_TBINS_v1.json](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/BORN_TBINS_v1.json) · result
- [FEATURES_v1.json](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/FEATURES_v1.json) · result
- [HARVEST_v1.json](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/HARVEST_v1.json) · result
- [RTT_HARVEST_v1.json](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/RTT_HARVEST_v1.json) · result
- [UNIT_ERRORS_v1.json](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/UNIT_ERRORS_v1.json) · result
- [WEAK_DIAGNOSTIC_v1.json](../2026-10-16-final-error-concentrated-late-near-threshold-births-dead-unit/WEAK_DIAGNOSTIC_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-cross-half-truth-product-removes-monte-carlo-noise-that) · [Copy provenance](../../../PROVENANCE.json)

