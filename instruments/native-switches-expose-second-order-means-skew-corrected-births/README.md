# Native switches expose second-order means, skew-corrected births and corrected transport gates

2026-10-02T18:06Z · tool · informational · measured

The candidate builder gains separate switches for the second-order mean term, the skew-corrected birth weight and transport-gate corrections. Native parity verifies the mean term to cosine 0.96-0.98 at layers 2-14 and the birth correction to 0.97 at its first affected layer; fixed gate consumers must also be updated consistently.

Recorded measurement:

    mean term +1.01% native vs +1.19% research MSE, 10 requests/layer, about 1 Mflop; birth -0.72% vs -1.01%, first affected-layer cosine 0.97 and size ratio 0.99; transport gate skew+kurtosis -0.51% vs -0.40%; collective feed -0.74% vs -0.67%

## Programs

- [code/fragment_so2_v1.py](code/fragment_so2_v1.py)
- [code/fragment_bsk_v1.py](code/fragment_bsk_v1.py)
- [code/fragment_gsk_v1.py](code/fragment_gsk_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L411](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L411)
- [TEAM_LOG_2026-10-01_to_10-22.md#L412](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L412)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-02-native-switches-expose-second-order-means-skew-corrected-births) · [Copy provenance](../../PROVENANCE.json)

