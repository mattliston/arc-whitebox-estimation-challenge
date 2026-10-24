# Gaussian low-rank factor plus constant residual equals Gaussian recursion; Q-node mixtures cost (2+68Q)n^3

2026-09-07 · finding · rejected · derived

A source audit showed that the state h = m + U z + epsilon with independent Gaussian z and a constant full residual covariance R is a single Gaussian whose ReLU mean and covariance cannot depend on the U/R split, so re-fitting that state after each ReLU adds no information; only a nonlinear conditional mean M(z) with a z-dependent conditional covariance R(z) supplies the missing mixed third-cumulant term E[d_i R_jk(z)]. A fixed-node realization is a Q-component Gaussian-mixture moment recursion whose GEMM/Cholesky subtotal is (2+68Q) n^3, and the earlier adaptive GMM4 prototype had already failed far from raw parity, so no run was proposed.

Recorded measurement:

    Q=4: 294.205B (utilization .133789, raw needed <7.474e-9 for adjusted <1e-9); Q=8: 586.263B (.266602, raw <3.751e-9). GMM4 provisional envelope 324.194B (.147427); GMM4 raw35/100 2.61823e-6/4.25983e-6 vs matched Gaussian 2.89618e-6/4.63005e-6.

## Evidence

- [RESIDUAL_PRESERVING_GAUSSIAN_FACTOR_SOURCE_AUDIT_V1.md#L9](RESIDUAL_PRESERVING_GAUSSIAN_FACTOR_SOURCE_AUDIT_V1.md#L9)
- [RESIDUAL_PRESERVING_GAUSSIAN_FACTOR_SOURCE_AUDIT_V1.md#L167](RESIDUAL_PRESERVING_GAUSSIAN_FACTOR_SOURCE_AUDIT_V1.md#L167)

<details>
<summary>All result files and records (1)</summary>

- [RESIDUAL_PRESERVING_GAUSSIAN_FACTOR_SOURCE_AUDIT_V1.md](RESIDUAL_PRESERVING_GAUSSIAN_FACTOR_SOURCE_AUDIT_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-gaussian-low-rank-factor-plus-constant-residual-equals-gaussian) · [Copy provenance](../../../PROVENANCE.json)

