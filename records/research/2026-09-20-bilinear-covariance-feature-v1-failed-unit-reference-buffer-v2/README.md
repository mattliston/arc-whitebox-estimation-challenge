# Bilinear covariance feature V1 failed its unit reference buffer, V2 its positive cold-diagonal domain (no value test)

2026-09-20 · incident · failed · measured

The first version of the terminal latent covariance feature study failed a read-only NumPy reference buffer in its unit harness before any feature-value comparison; the narrowly repaired second version then failed its declared all-diagonals-positive domain because the cold covariance diagonals on nets 35/100 contain negative entries while active diagonals are positive. Neither version ran a mean-value test; a third version was separately specified with literal signed cold products.

Recorded measurement:

    negative values only in cold diagonals on nets 35/100; active diagonals all positive

## Evidence

- [NEGATIVE_RESULT.md#L42](NEGATIVE_RESULT.md#L42)
- [NEGATIVE_RESULT.md#L43](NEGATIVE_RESULT.md#L43)
- [NEGATIVE_RESULT.md#L44](NEGATIVE_RESULT.md#L44)
- [NEGATIVE_RESULT.md#L45](NEGATIVE_RESULT.md#L45)

<details>
<summary>All result files and records (5)</summary>

- [REPAIR.md](REPAIR.md) · record
- [domain_result.json](domain_result.json) · result
- [closeout.json](closeout.json) · record
- [SIGNED_COLD_V3.md](SIGNED_COLD_V3.md) · record
- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-bilinear-covariance-feature-v1-failed-unit-reference-buffer-v2) · [Copy provenance](../../../PROVENANCE.json)

