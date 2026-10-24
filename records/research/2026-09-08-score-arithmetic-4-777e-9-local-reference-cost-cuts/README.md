# Score arithmetic from the 4.777e-9 local reference: cost cuts alone cannot reach 3e-9, a raw-error route is mandatory

2026-09-08T07:19Z · decision · adopted · derived

From the qualified local reference (independent14 adjusted 4.7768418485455745e-9, noise-corrected raw 2.2458484596872894e-8, mean complete bill 466705080466.9286 FLOPs) a uniform complete-cost reduction at fixed raw error would have to exceed 16.2627% to cross 4e-9 and 37.1970% to cross 3e-9, and at the 0.1 utilisation floor the present raw error would still score 2.24585e-9. The record concludes that a raw-error route is necessary, not optional; the objective stays official adjusted strictly below 1e-9 with milestone banks below 4e-9, 3e-9 and 2e-9.

Recorded measurement:

    cost ratios needed at fixed raw: < .83737334 (4e-9) and < .62803000 (3e-9); at the .1 multiplier floor the current raw gives 2.24585e-9; from the later 409f903e reference the ratios are < .8385330357 / < .6288997768 and below 1e-9 needs more than 55.47339823% raw reduction even at the floor

## Evidence

- [SOURCE_DECISIONS_2026_09_08_V1.md#L19](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md#L19)
- [SOURCE_DECISIONS_2026_09_08_V1.md#L22](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md#L22)
- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L52](SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md#L52)
- [JOINT_CORE_AND_BINI_ERROR_SOURCE_AUDIT_2026_09_08_V1.md#L38](JOINT_CORE_AND_BINI_ERROR_SOURCE_AUDIT_2026_09_08_V1.md#L38)

<details>
<summary>All result files and records (3)</summary>

- [SOURCE_DECISIONS_2026_09_08_V1.md](../2026-09-08-moving-unsquared-query-leg-q-w-t-p-keeps/SOURCE_DECISIONS_2026_09_08_V1.md) · record
- [SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md](SUPPORT_QUERY_AND_APA_PREFLIGHT_2026_09_08_V1.md) · record
- [JOINT_CORE_AND_BINI_ERROR_SOURCE_AUDIT_2026_09_08_V1.md](JOINT_CORE_AND_BINI_ERROR_SOURCE_AUDIT_2026_09_08_V1.md) · record

</details>

## Related work

- ← measured by: [Removing four unused materializations saves 646348799 FLOPs per network and improves adjusted error 0.13830087084589%](../../../experiments/flop-accounting/removing-four-unused-materializations-saves-646348799-flops-per-network/README.md)

[Research graph](../../../graph/index.html#node=2026-09-08-score-arithmetic-4-777e-9-local-reference-cost-cuts) · [Copy provenance](../../../PROVENANCE.json)

