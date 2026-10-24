# Merged newborn-common product parked: warm FLOPs -0.148% on one network and +0.083% on another

2026-10-12T18:41:06Z · experiment · rejected · measured

Priced exactly through the file's own matmul wrapper, merging the newborn-common push product into the young push product (B = [p | base[prev]]) saves 0.160% of metered FLOPs summed over layers 2-15 (minus a 0.033% concatenate) at rounding level; measured on the candidate generation 50 (S50) twin with every prebind key bound, the warm pass gave -0.148% metered FLOPs (-610 charged ops) on 7100136 but the cold first network +0.001% (the merged products' new tree keys bind lazily) and 7100137 +0.083% warm (+0.31% cold), because the merged width changes the push's 2048-column packet split; the lever was parked pending a per-layer merge decision from the planner's price model.

Recorded measurement:

    priced -0.160% over L2-15 (0 at L2-3, +0.045% at L5-6), concat 0.033%, ops -800; v1 warm -0.039%, -672 ops (L7-15 only); v2 warm 7100136 -0.148% (258,311,456,554 -> 257,929,530,645), ops -610, cold +0.001%, prediction change rms 2.1e-6; 7100137 warm +0.083%, cold +0.31%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/scout_native/price_nbmerge_v1.py](code/scout_native/price_nbmerge_v1.py)
- [code/scout_native/fragment_nbm_v2.py](code/scout_native/fragment_nbm_v2.py)
- [code/scout_native/run_cert_v2.py](code/scout_native/run_cert_v2.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/engineer/a62_modules/s49/jmS50_v1tw_x/candidate.py](code/engineer/a62_modules/s49/jmS50_v1tw_x/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1608](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1608)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1618](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1618)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1629](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1629)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1632](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1632)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1633](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1633)

<details>
<summary>All result files and records (2)</summary>

- [price_nbmerge_s50tw.json](price_nbmerge_s50tw.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-12-merged-newborn-common-product-parked-warm-flops-0-148) · [Copy provenance](../../../PROVENANCE.json)

