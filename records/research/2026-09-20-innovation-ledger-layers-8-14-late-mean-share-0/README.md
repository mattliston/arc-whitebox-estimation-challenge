# Innovation ledger at layers 8-14: late mean (share 0.388) and off-diagonal covariance (0.336) dominate the final error

2026-09-20 · finding · informational · measured

Each true-reference innovation d_l = x_l - T_l x_(l-1) of the lazy-rotation card was split into mean, diagonal-covariance and off-diagonal-covariance parts and propagated to the final error through the fixed Gaussian-gain tangent on net 35 (cross-products of two independent 8M reference streams, total cross-error energy 1.8212163540060948e-8). Late mean innovations and late off-diagonal covariance innovations carry most of the endpoint error while diagonal covariance innovations contribute almost nothing, which narrowed the next target to a causal supplier of late mean and full off-diagonal covariance information.

Recorded measurement:

    net 35 signed alignment share / formal-removal residual ratio: initial_mean -0.012607 / 0.984800; initial_cov_diag -0.001736 / 1.005794; initial_cov_offdiag 0.165520 / 0.769124; innovation_mean 0.388082 / 0.672932; innovation_cov_diag -0.002855 / 1.007046; innovation_cov_offdiag 0.335738 / 0.729961; final_mean_defect 0.127859 / 0.879749; net 100 (one stream) ratios 0.730222, 0.995265, 0.875585, 0.845851, 0.998432, 0.770312, 0.905631; initial-7 mean cross energy -7.360395661737827e-10 vs observed 5.649297881802429e-9 and replica difference 6.38533744797621e-9

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L12](RESULT.md#L12)
- [RESULT.md#L15](RESULT.md#L15)
- [RESULT.md#L17](RESULT.md#L17)
- [RESULT.md#L21](RESULT.md#L21)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](RESULT.md) · result
- [result.json](result.json) · result
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-innovation-ledger-layers-8-14-late-mean-share-0) · [Copy provenance](../../../PROVENANCE.json)

