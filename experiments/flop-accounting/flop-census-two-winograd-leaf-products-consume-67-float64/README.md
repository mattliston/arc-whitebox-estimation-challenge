# FLOP census: two Winograd leaf products consume 67%; float64-rate work is only 0.56%

2026-10-12T16:01Z · experiment · rejected · measured

Every charge of the submitted twin attributed to its site: the two Winograd leaf batched-matmul sites are 67% and tree additions ~15%; float64-rate work is 0.56% (ceiling -0.28%, realistic -0.05%), symmetric products are exploited, and L15 already runs the minimal final path; the clean non-retraining cut is f32 zero fills (-0.03%).

Recorded measurement:

    _hz_leaf 37.0%, _ipt_leaf_unwind 29.9%, k-split 4.7%; by step push 44.4%, read 35.3%, dead-cut star 9.1%, L15 k3 3.5%; by layer L7 9.02% max, L15 3.92%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/engineer/a62_modules/census_bill_v1.py](code/engineer/a62_modules/census_bill_v1.py)
- [code/engineer/a62_modules/census_layer_v1.py](code/engineer/a62_modules/census_layer_v1.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/jmS50_v1tw_x/candidate.py](code/engineer/a62_modules/s49/jmS50_v1tw_x/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1495](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1495)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1514](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1514)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1517](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1517)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1793](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1793)

<details>
<summary>All result files and records (3)</summary>

- [census_bill_s50tw.json](census_bill_s50tw.json) · result
- [census_layer_s50tw.json](census_layer_s50tw.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [FLOP census: exact kernel changes each save at most 0.2%; live-floor push cut adds 13% error](../flop-census-exact-kernel-changes-each-save-most-0/README.md)
- enables: [Float32 zero fills and cached Winograd transforms adopted: about -0.24% FLOPs, byte-identical](../../../records/research/2026-10-12-float32-zero-fills-cached-winograd-transforms-adopted-about-0/README.md)
- enables: [Identity caching and repeated-row cuts adopted; float32 substitution rejected because rounding exceeds savings](../../../records/research/2026-10-13-identity-caching-repeated-row-cuts-adopted-float32-substitution-rejected/README.md)
- enables: [Reassociated push/read algebra adopted: -0.195% FLOPs at rounding-level prediction change](../../../records/research/2026-10-13-reassociated-push-read-algebra-adopted-0-195-flops-rounding/README.md)

[Research graph](../../../graph/index.html#node=2026-10-12-flop-census-two-winograd-leaf-products-consume-67-float64) · [Copy provenance](../../../PROVENANCE.json)

