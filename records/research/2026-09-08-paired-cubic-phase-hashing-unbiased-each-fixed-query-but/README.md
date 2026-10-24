# Paired cubic phase hashing is unbiased for each fixed query but has relative noise about 11/3 at break-even buckets

2026-09-08T09:32Z · finding · rejected · derived

Hashing the signed factor pairs of the third bank into B buckets with independent cube-root-of-unity phases gives a real estimator Z of the cubic query sum (x_r^3 - y_r^3) that is unbiased for a fixed query, with exact variance 9/(2B) sum_{r<s} (A_rs + A_sr)^2 + 18/B^2 sum_{r<s<t} D_rst^2 evaluable in O(R) power sums. In a declared Gaussian-column reference with within-pair correlation tending to one the ratio of expected noise to expected signal tends to (7/6)(R-1)/B + (1/3)(R-1)(R-2)/B^2, i.e. about 11/3 at the generic inner-width break-even B = R/2 and 10 at B = R/4; the push and reader need four B-wide real channels, and the card's nonlinear feedback destroys the fixed-query unbiasedness, so no bucket, seed or rank run was admitted.

Recorded measurement:

    noise/signal ratio of expectations -> 11/3 at B ~ R/2, 10 at B ~ R/4 (reference model); equal-column example Var Z / T^2 = 9(R-1)/(BR) + 3(R-1)(R-2)/(B^2 R); generic affine push inner width 4B vs parent 2R

## Evidence

- [PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md#L41](PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md#L41)
- [PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md#L86](PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md#L86)
- [PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md#L92](PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md#L92)
- [PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md#L99](PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md#L99)
- [PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md#L119](PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md#L119)

<details>
<summary>All result files and records (1)</summary>

- [PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md](PAIRED_CUBIC_HASH_SOURCE_AUDIT_2026_09_08_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-paired-cubic-phase-hashing-unbiased-each-fixed-query-but) · [Copy provenance](../../../PROVENANCE.json)

