# Two unimplemented exact micro-cuts (denominator reuse, dead terminal covariance assignment): 70,284,316 F (0.0151%)

2026-09-08T10:25Z · finding · inconclusive · derived

Source inspection of estimator 409f903e found the post-covariance metric denominator trace(C)^2 + 2 sum(C*C) computed in the pair map and again in the caller before any mutation of C on layers 1 to 14, and the layer-15 consistent covariance update C - outer(m,dm) - outer(dm,m) - outer(dm,dm) assigned after the learned hook and never consumed. After an owner-aware tariff correction (symmetric self-outer, unwrapped trace scalar) the two deltas are 58,748,956 F and 11,535,360 F; both were held unimplemented for a later combined edit and never counted as gains.

Recorded measurement:

    denominator reuse 4 n^2 + 2 n + 2 per occurrence, 14 occurrences = 58,748,956 F (first priced 58,749,404 F); dead assignment 11 n^2 + n = 11,535,360 F (first priced 12 n^2 = 12,582,912 F); sum 70,284,316 F = .01508057059% of 466058731667.9286 F; conditional adjusted 4.769517625422612e-9 vs qualified 4.770235434670105e-9

## Evidence

- [DENOMINATOR_REUSE_HELD_COMPONENT_2026_09_08_V1.md#L11](DENOMINATOR_REUSE_HELD_COMPONENT_2026_09_08_V1.md#L11)
- [DENOMINATOR_REUSE_HELD_COMPONENT_2026_09_08_V1.md#L31](DENOMINATOR_REUSE_HELD_COMPONENT_2026_09_08_V1.md#L31)
- [FINAL_COV_DEAD_ASSIGNMENT_HELD_COMPONENT_2026_09_08_V1.md#L9](FINAL_COV_DEAD_ASSIGNMENT_HELD_COMPONENT_2026_09_08_V1.md#L9)
- [HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md#L32](HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md#L32)
- [HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md#L53](HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md#L53)
- [HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md#L58](HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md#L58)

<details>
<summary>All result files and records (3)</summary>

- [DENOMINATOR_REUSE_HELD_COMPONENT_2026_09_08_V1.md](DENOMINATOR_REUSE_HELD_COMPONENT_2026_09_08_V1.md) · record
- [FINAL_COV_DEAD_ASSIGNMENT_HELD_COMPONENT_2026_09_08_V1.md](FINAL_COV_DEAD_ASSIGNMENT_HELD_COMPONENT_2026_09_08_V1.md) · record
- [HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md](HELD_MICRO_COMPONENT_TARIFF_CORRECTION_2026_09_08_V1.md) · record

</details>

## Related work

- enables: [Removing four unused materializations saves 646348799 FLOPs per network and improves adjusted error 0.13830087084589%](../../../experiments/flop-accounting/removing-four-unused-materializations-saves-646348799-flops-per-network/README.md)

[Research graph](../../../graph/index.html#node=2026-09-08-two-unimplemented-exact-micro-cuts-denominator-reuse-dead-terminal) · [Copy provenance](../../../PROVENANCE.json)

