# Corrected annealed kernel law; original numerical failures retained

2026-09-06T09:39Z · finding · informational · measured

Two Cauchy FFT grids/radii independently recover the low64 coefficients to1.471e-12 absolute agreement. Degree1 agrees with an independent chain- derivative calculation to floating-point precision; coefficients are real and nonnegative at the tested resolution. The archived empirical sampling risks, low-degree capture measurements, and tested Monte Carlo (MC)/affine-CV/antithetic/lattice/subspace failures are unchanged.

Recorded measurement:

    Two Cauchy FFT grids/radii independently recover the low64 coefficients to1.471e-12 absolute agreement. Degree1 agrees with an independent chain- derivative calculation to floating-point precision; coefficients are real and nonnegative at the tested resolution. At depth16, corrected cumulative mass through degrees2/4/8/16/32/64 is .341632/.524462/.704019/.842043/.926548/.969303. Mean degree is14.1756, not the old approximately460. The correct raw-map branching process has offspring0 and mean1; conditioning on nonzero final degree happens at the END. The old no-zero-offspring window argument does not transfer. Primary raw-kernel reference: https://arxiv.org/abs/1711.00165 (Lee et al., Deep Neural Networks as Gaussian Processes); implementation corroboration: https://github.com/google/neural-tangents/blob/main/neural_tangents/_src/stax/elementwise.py .

## Programs

- [audit_nngp_centering_v1.py](audit_nngp_centering_v1.py)
- [../../../experiments/flop-accounting/deterministic-quadrature-route-measured-rejected/code/theory_branching.py](../../../experiments/flop-accounting/deterministic-quadrature-route-measured-rejected/code/theory_branching.py)

## Evidence

- [NNGP_CENTERING_AUDIT_V1_CORRECTION.md#L1](NNGP_CENTERING_AUDIT_V1_CORRECTION.md#L1)
- [NNGP_CENTERING_AUDIT_V1_CORRECTION.md#L46](NNGP_CENTERING_AUDIT_V1_CORRECTION.md#L46)

<details>
<summary>All result files and records (3)</summary>

- [NNGP_CENTERING_AUDIT_V1_CORRECTION.md](NNGP_CENTERING_AUDIT_V1_CORRECTION.md) · record
- [nngp_centering_audit_v1.json](nngp_centering_audit_v1.json) · result
- [NNGP_CENTERING_AUDIT_DESIGN_V1.md](NNGP_CENTERING_AUDIT_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-corrected-annealed-kernel-law-original-numerical-failures-retained) · [Copy provenance](../../../PROVENANCE.json)

