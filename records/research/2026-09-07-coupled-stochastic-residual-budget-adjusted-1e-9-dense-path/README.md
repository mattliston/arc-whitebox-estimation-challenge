# Coupled stochastic residual budget: adjusted <1e-9 at dense path cost needs residual variance <6.5536e-5 (99.9% capture)

2026-09-07T00:22Z · finding · rejected · derived

For a samplewise surrogate s(X) with known mean, the adjusted score is (v_R/N) max(0.1, (C0+N c)/2^41), so above the floor the sample term is v_R (c + C0/N)/2^41 and more paths cannot overcome an excessive variance times path-cost product. Under the declared classical dense convention c = 2^25 FLOPs per 16-layer path, adjusted below 1e-9 requires mean residual variance below 6.5536e-5, i.e. capture above 99.9019% of the archived rank-35 final activation variance .06679448554. The audit re-scoped every executed sampling arm (affine control capture .20505, GH2, lattice, four-Walsh witness on HOLD, Hermite degree-12 Gaussian residual 4.41320e-5) and proved that any zero-bias shallow ridge control has an exactly linear odd part, so it cannot cancel odd Hermite degrees >= 3 (fitted odd mass .566128 vs linear mass .224295). No new control was identified; no sampling run was launched.

Recorded measurement:

    v_R < 6.5536e-5 for adjusted <1e-9 at c=2^25 per path; one extra dense control application tightens it to 6.16809e-5; with 160 samples raw <1e-8 at the .1 floor needs v_R < 1.6e-6 (99.9976% capture). Prior: MC MSE 1.8942e-5, affine 1.4422e-5, antithetic 1.5834e-5 at 4096 paths; gradient-basis conditional captures 6.5% at 8, 27.7% at 128, 63.1% at 512 directions.

## Evidence

- [COUPLED_STOCHASTIC_RESIDUAL_SOURCE_AUDIT_V1.md#L42](COUPLED_STOCHASTIC_RESIDUAL_SOURCE_AUDIT_V1.md#L42)
- [COUPLED_STOCHASTIC_RESIDUAL_SOURCE_AUDIT_V1.md#L74](COUPLED_STOCHASTIC_RESIDUAL_SOURCE_AUDIT_V1.md#L74)
- [COUPLED_STOCHASTIC_RESIDUAL_SOURCE_AUDIT_V1.md#L140](COUPLED_STOCHASTIC_RESIDUAL_SOURCE_AUDIT_V1.md#L140)

<details>
<summary>All result files and records (1)</summary>

- [COUPLED_STOCHASTIC_RESIDUAL_SOURCE_AUDIT_V1.md](COUPLED_STOCHASTIC_RESIDUAL_SOURCE_AUDIT_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-coupled-stochastic-residual-budget-adjusted-1e-9-dense-path) · [Copy provenance](../../../PROVENANCE.json)

