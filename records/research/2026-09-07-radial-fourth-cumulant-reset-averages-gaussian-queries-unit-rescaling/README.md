# Radial fourth-cumulant reset averages Gaussian queries; unit-rescaling non-invariance is expected

2026-09-07T00:46Z · finding · informational · derived

The card's scalar radial fourth-order ansatz Q_rad = (r/3)(C C + C C + C C) with r = 3 sum_ij Q_iijj / [(tr C)^2 + 2 ||C||_F^2] was shown by a Wick contraction to equal E_w q(w) / E_w v(w)^2 for an isotropic Gaussian query w, not E_w[q/v^2] and not a least-squares fit. Under a diagonal query metric A the scalar r_A is exactly invariant to positive per-unit rescaling, so keeping an isotropic metric after rescaling changes the projection objective rather than exposing an implementation error; a standardized-radial replacement was therefore not adopted as a bug fix. The audit also recorded that the code does not symmetrize every covariance push, so a skew part K inflates the denominator by 2 ||K||_F^2.

## Evidence

- [GAUGE_RADIAL_PROJECTION_INTERPRETATION_V1.md#L9](GAUGE_RADIAL_PROJECTION_INTERPRETATION_V1.md#L9)
- [GAUGE_RADIAL_PROJECTION_INTERPRETATION_V1.md#L40](GAUGE_RADIAL_PROJECTION_INTERPRETATION_V1.md#L40)
- [GAUGE_RADIAL_PROJECTION_INTERPRETATION_V1.md#L55](GAUGE_RADIAL_PROJECTION_INTERPRETATION_V1.md#L55)

<details>
<summary>All result files and records (1)</summary>

- [GAUGE_RADIAL_PROJECTION_INTERPRETATION_V1.md](GAUGE_RADIAL_PROJECTION_INTERPRETATION_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-radial-fourth-cumulant-reset-averages-gaussian-queries-unit-rescaling) · [Copy provenance](../../../PROVENANCE.json)

