# Terminal row sensitivity cannot beat the standardized mean threshold of 3

2026-10-19T16:13:38Z · finding · informational · derived

The scored output is the final post-activation mean m = mg + k3 e/6 + k4 f/24 with e = -z phi(z)/v, so a row's k3 sensitivity is -z phi(z)/(6v): zero at the kink and beyond |z| of about 3, peaking at |z| = 1. An oracle price list with the post-push k3 is cheap (dropping the 10% cheapest kept rows' k3 adds +0.15% of MSE) but needs the k3 the push itself computes; ranking rows by the pre-push coefficient alone costs +1.2% at the first 5%, and a two-pass estimate of k3 from the heaviest 5/10/20% of pairs costs +19.6/+18.1/+10.3% because a row's k3 is a sum of many nearly cancelling pair terms. The fired z 3.0 row cut is optimal among realizable rules; the kink-side arms were pulled.

Recorded measurement:

    max rel diff 1.8e-4 between the formula and the predict output; oracle 10% cheapest kept rows -> +0.15% MSE (the fired 3<|z|<=3.5 band +0.10% for 86 rows); pre-push ranking +1.2% for the first 5%; oracle pair cut old pairs 10% -> +0.11%, 20% -> +0.6%; two-pass (5/10/20% of pairs) +19.6/+18.1/+10.3% for the cheapest 10% of rows

## Programs

- [../2026-10-19-final-layer-cuts-help-only-where-third-cumulant-sensitivity/probe_l15sens_v4.py](../2026-10-19-final-layer-cuts-help-only-where-third-cumulant-sensitivity/probe_l15sens_v4.py)
- [probe_l15sens_v5.py](probe_l15sens_v5.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5975](../../TEAM_LOG_2026-10-01_to_10-22.md#L5975)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5978](../../TEAM_LOG_2026-10-01_to_10-22.md#L5978)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5981](../../TEAM_LOG_2026-10-01_to_10-22.md#L5981)

<details>
<summary>All result files and records (3)</summary>

- [l15sens_v4_rel_l14x_7103256.json](l15sens_v4_rel_l14x_7103256.json) · result
- [l15sens_v5_rel_l14x_7103256.json](l15sens_v5_rel_l14x_7103256.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-terminal-row-sensitivity-cannot-beat-standardized-mean-threshold-3) · [Copy provenance](../../../PROVENANCE.json)

