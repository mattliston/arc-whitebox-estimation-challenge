# Complete signed-pair cubic kernel reduces leading Gram work 25% and requires a 512-dimensional solve

2026-09-07T09:04Z · finding · informational · derived

To avoid discarding a signed pair's difference direction, each dictionary atom was redefined as the complete unweighted pair tensor T = p^3 - q^3 = 3 Sym(H,H,D) + D^3/4 with H = (p+q)/2 and D = p-q; its full, repeated-plane and ordered all-distinct inner products were derived in closed form (K_distinct = K_full - 3 K_plane + 2 t.t') and verified independently. Nine ordinary FP32 cross-Grams replace the three physical-atom Grams, 25% lower at matched physical width, with a 512-dimensional instead of 1024-dimensional solve, but no capture or endpoint-value evidence existed.

Recorded measurement:

    Leading Gram bill 9 (2 a K_pair M_pair - K_pair M_pair) = 25% below the three physical cross-Grams; solve dimension 512 vs 1024 at 1024 retained physical columns.

## Evidence

- [COMPLETE_PAIR_KERNEL_SOURCE_V1.md#L18](COMPLETE_PAIR_KERNEL_SOURCE_V1.md#L18)
- [COMPLETE_PAIR_KERNEL_SOURCE_V1.md#L44](COMPLETE_PAIR_KERNEL_SOURCE_V1.md#L44)

<details>
<summary>All result files and records (1)</summary>

- [COMPLETE_PAIR_KERNEL_SOURCE_V1.md](COMPLETE_PAIR_KERNEL_SOURCE_V1.md) · record

</details>

## Related work

- enables: [Whole-pair global projection after layer 7: 512 pairs cut compute 19.03% but increase raw error 6.06x](../../../experiments/third-order-planes/whole-pair-global-projection-after-layer-7-512-pairs/README.md)

[Research graph](../../../graph/index.html#node=2026-09-07-complete-signed-pair-cubic-kernel-reduces-leading-gram-work) · [Copy provenance](../../../PROVENANCE.json)

