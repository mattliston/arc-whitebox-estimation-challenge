# Billed-view census: basic indexing free, reshape bills its input size (0.69% warm), lazy tree binds 0.29% warm

2026-10-16T16:18:14Z · finding · informational · measured

A per-site charge probe on the S3-class build showed flopscope bills basic indexing at zero (views) and getitem only for advanced gathers (4 flops per element, 0.17% warm, all data-dependent), so view rewrites save requests but no bill. reshape bills numel of its input every time and no free op splits an axis; the fresh-operand reshapes of the tree kernels are inherent. The one movable item was the lazy tree binds in _ipt_view (24 views, 754M elements), of which 21 lay outside the boxes of lazybind v2.

Recorded measurement:

    Advanced gathers 0.17% warm (_xsg_rows A[idx] .046%, bd_base32[:, selected] .019%, table gathers ~.01% each); reshape .69% warm / 1.02% cold, copyto .35%; fresh-operand reshapes _l3o_last .08%, _rs_load_a/b .13% + copyto .13%, _hzd_load .05% + .05%; lazy tree binds in _ipt_view .29% warm / .27% cold.

## Programs

- [build_billprobe_v1.py](build_billprobe_v1.py)
- [multi_judge_v1.py](multi_judge_v1.py)
- [cand_s3ish_billprobe_v1.py](cand_s3ish_billprobe_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [cand_s1bp5_g0_v3.py](cand_s1bp5_g0_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4326](../../TEAM_LOG_2026-10-01_to_10-22.md#L4326)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Lazy-bind v4 contiguous-leaf boxes: -0.08% bill for +1-5% server instructions and +0.1-4.6% client residual, dropped](../../../experiments/flop-accounting/lazy-bind-v4-contiguous-leaf-boxes-0-08-bill/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-billed-view-census-basic-indexing-free-reshape-bills-input) · [Copy provenance](../../../PROVENANCE.json)

