# Layer-13 four-cycle mean diagnostic uses one FP64 matrix-vector product, about 4,194,304 FLOPs

2026-09-07T10:14Z · finding · informational · derived

A provenance audit corrected the earlier statement that no matching error reference existed: the sealed rank-35 capture of the official bank (all 16 served outputs, direction u and Rayleigh value theta from proxy_values.npz, the literal bank-Q(M) = trace(M)^2 + 2 sum(M*M), the sealed omega = (alpha13^2-1) pdf13 / (24 pre_v13 sigma13)) together with the existing 1e9 true post means at layer 13 allow a direct mean-direction diagnostic delta_m13 = omega (3 a^4 - rho pre_v13^2) with a = W13.T f, f = theta dA u^2 and rho = 9 (sum f^2)^2 / bank_Q, costing one FP64 1024x1024 matrix-vector product. The check had not been run and admitted no carrier.

Recorded measurement:

    About 4,194,304 metered FLOPs plus O(1024) vector work; reference bank capture 490,466,018,197 FLOPs, 21.0457798336 s, peak 6,117,900,288 bytes.

## Evidence

- [GAUSSIAN_FOUR_CYCLE_EXISTING_MEAN_WITNESS_V1.md#L8](GAUSSIAN_FOUR_CYCLE_EXISTING_MEAN_WITNESS_V1.md#L8)
- [GAUSSIAN_FOUR_CYCLE_EXISTING_MEAN_WITNESS_V1.md#L62](GAUSSIAN_FOUR_CYCLE_EXISTING_MEAN_WITNESS_V1.md#L62)
- [GAUSSIAN_FOUR_CYCLE_EXISTING_MEAN_WITNESS_V1.md#L104](GAUSSIAN_FOUR_CYCLE_EXISTING_MEAN_WITNESS_V1.md#L104)

<details>
<summary>All result files and records (1)</summary>

- [GAUSSIAN_FOUR_CYCLE_EXISTING_MEAN_WITNESS_V1.md](GAUSSIAN_FOUR_CYCLE_EXISTING_MEAN_WITNESS_V1.md) · record

</details>

## Related work

- ← enables: [Gaussian four-quadratic cycle source Q\[w^4\] = 3 tr((C diag(d w))^4) had no measured value](../2026-09-07-gaussian-four-quadratic-cycle-source-q-w-4-3/README.md)

[Research graph](../../../graph/index.html#node=2026-09-07-layer-13-four-cycle-mean-diagnostic-uses-one-fp64) · [Copy provenance](../../../PROVENANCE.json)

