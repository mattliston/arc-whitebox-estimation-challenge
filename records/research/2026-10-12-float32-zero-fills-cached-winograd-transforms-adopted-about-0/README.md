# Float32 zero fills and cached Winograd transforms adopted: about -0.24% FLOPs, byte-identical

2026-10-12T19:53Z · decision · adopted · derived

scout-native's byte-identical metered FLOPs package: float32 zero fills (-0.034%), the old tier's constant B-transforms kept from L10 (and the L7 recon trees for L8-9) across layers, and the newborn-common push reusing the young push's strip A tree; NB-MERGE was parked as network-dependent and GNN-W deferred.

Recorded measurement:

    TWIN 7100136 / 137 / 600 / 601 bill -0.2402 / -0.2419 / -0.2221 / -0.2444%, pred unchanged, ops -569 / -598; SAFE -0.2399 / -0.2416%; footprint ~2.8 GB; NB-MERGE warm -0.148% on 136 but +0.083% on 137

## Programs

- [fragment_f32z_v1.py](fragment_f32z_v1.py)
- [fragment_ubx_v2.py](fragment_ubx_v2.py)
- [fragment_sn3_v1.py](fragment_sn3_v1.py)
- [check_sn3_parent_v1.py](check_sn3_parent_v1.py)
- [../../../experiments/arithmetic-kernels/merged-newborn-common-product-parked-warm-flops-0-148/code/scout_native/fragment_nbm_v2.py](../../../experiments/arithmetic-kernels/merged-newborn-common-product-parked-warm-flops-0-148/code/scout_native/fragment_nbm_v2.py)
- [../../../experiments/arithmetic-kernels/winograd-head-message-products-parked-0-10-flops-costs/code/fragment_gnnw_v1.py](../../../experiments/arithmetic-kernels/winograd-head-message-products-parked-0-10-flops-costs/code/fragment_gnnw_v1.py)
- [dims_probe_v2.py](dims_probe_v2.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)
- [../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/jmS50_v1tw_x/candidate.py](../../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/jmS50_v1tw_x/candidate.py)
- [candidate.py](candidate.py)
- [../../../experiments/arithmetic-kernels/winograd-head-message-products-parked-0-10-flops-costs/code/sn3_s50tw/candidate.py](../../../experiments/arithmetic-kernels/winograd-head-message-products-parked-0-10-flops-costs/code/sn3_s50tw/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1573](../../TEAM_LOG_2026-10-01_to_10-22.md#L1573)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1608](../../TEAM_LOG_2026-10-01_to_10-22.md#L1608)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1618](../../TEAM_LOG_2026-10-01_to_10-22.md#L1618)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1629](../../TEAM_LOG_2026-10-01_to_10-22.md#L1629)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1632](../../TEAM_LOG_2026-10-01_to_10-22.md#L1632)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1639](../../TEAM_LOG_2026-10-01_to_10-22.md#L1639)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1643](../../TEAM_LOG_2026-10-01_to_10-22.md#L1643)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1645](../../TEAM_LOG_2026-10-01_to_10-22.md#L1645)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1650](../../TEAM_LOG_2026-10-01_to_10-22.md#L1650)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1652](../../TEAM_LOG_2026-10-01_to_10-22.md#L1652)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1654](../../TEAM_LOG_2026-10-01_to_10-22.md#L1654)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1655](../../TEAM_LOG_2026-10-01_to_10-22.md#L1655)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1657](../../TEAM_LOG_2026-10-01_to_10-22.md#L1657)

<details>
<summary>All result files and records (3)</summary>

- [cert_sn3tw.json](cert_sn3tw.json) · result
- [cert_sn3s.json](cert_sn3s.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #335633: corrected transport gate plus byte-identical transform caching](../../../submissions/335633/README.md)
- ← enables: [FLOP census: two Winograd leaf products consume 67%; float64-rate work is only 0.56%](../../../experiments/flop-accounting/flop-census-two-winograd-leaf-products-consume-67-float64/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-float32-zero-fills-cached-winograd-transforms-adopted-about-0) · [Copy provenance](../../../PROVENANCE.json)

