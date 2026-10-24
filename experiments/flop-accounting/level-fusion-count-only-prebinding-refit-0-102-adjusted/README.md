# Level fusion and count-only prebinding refit: -0.102% adjusted versus #337028 on 32 paired networks

2026-10-19T06:24:38Z · experiment · adopted · measured

The fusion file eab36c88 (level fusion of product sides, 'fusion18') plus a count-only in-place prebind refit with same-size swaps is byte-identical to the fusion file and became the stacking base: -0.358% adjusted vs the 95c control on seven ranks, -0.269% on 32 ranks, and paired directly against the fired bundle -0.102% adjusted (bill -0.092%, raw -0.009%). Calibration: the seven ranks overstate this lineage by about 0.09 points and local 32-rank reads are conservative vs the platform by about 0.06 points.

Recorded measurement:

    a7ffd713 7 ranks vs 95c: raw -0.112% (se .11), bill -0.247%, adjusted -0.358% (se .13); requests -667/net warm, -855 at rank 35; residual x0.979; max RSS 7.04 GB (bundle 7.46); 32 ranks vs 95c: all32 -0.269% (se .079, 21/32 lower; raw ~-0.03%, bill -0.24%), heldout25 -0.244% (se .095), seven -0.358%; base minus bundle on 32: adjusted -0.102% (se .029, 25/32 lower), raw -0.009%, bill -0.092%; heldout25 -0.099% (se .036), seven -0.110% (7/7); bundle -0.168% vs 95c locally on 32 vs platform -0.229%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_oct19_fusion_pb_v1/candidate_fusion_pbv1_v2.py](code/fable_oct19_fusion_pb_v1/candidate_fusion_pbv1_v2.py)
- [code/fable_oct19_half_v1/compare32_v1.py](code/fable_oct19_half_v1/compare32_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5905](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5905)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5913](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5913)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5916](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5916)

<details>
<summary>All result files and records (4)</summary>

- [science7_v2.log](science7_v2.log) · log
- [science32_base_sci.log](science32_base_sci.log) · log
- [science32_bundle.log](science32_bundle.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #337061: level fusion, prebinding refit and two operand box views, graded 1.4427189e-9](../../../submissions/337061/README.md)

[Research graph](../../../graph/index.html#node=2026-10-19-level-fusion-count-only-prebinding-refit-0-102-adjusted) · [Copy provenance](../../../PROVENANCE.json)

