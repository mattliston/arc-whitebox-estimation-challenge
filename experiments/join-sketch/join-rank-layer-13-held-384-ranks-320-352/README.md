# Join rank at layer 13 held at 384: ranks 320, 352, 448 and 512 all worse adjusted (+0.50% to +0.97%)

2026-10-19T11:32:44Z · experiment · rejected · measured

The rank of the joined plane frame at layer 13 was moved below and above the shipped 384. Ranks 320 and 352 cut the bill but cost raw MSE (about +0.5% raw per 32 of rank), ranks 448 and 512 bought little raw for a growing bill (the old-tier products at layers 14-15 and the join grow with the frame), so the curve is a knee at 384; every join rank on this lineage (layers 7, 10, 13) is measured at 1:1 or worse.

Recorded measurement:

    320 (7 ranks vs 95c): raw +1.114% (se .42, 2/7 lower), bill -0.554%, adjusted +0.554% (= +0.80% vs #337028), requests +200..+440/net; 352 (32 ranks vs the base): adjusted +0.498% (se .15, 10/32 lower; raw +0.582%, bill -0.082%); 448 (28 ranks vs stk1): raw -0.113%, bill +0.644%, adjusted +0.532% (6/28 lower; held-out -0.178% / +0.648%); 512: raw -0.170%, bill +1.142%, adjusted +0.973%; part01 bill +0.60/+0.79/+0.60% (448) and +1.12/+1.29/+1.10% (512), requests +466..+977, max RSS 7.72 / 7.60 GB; earlier reads L7 +1.7% MSE, L10 +1.8% adjusted

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_oct19_l13rank_v1/candidate_jsym_L13rank320_v1.py](code/fable_oct19_l13rank_v1/candidate_jsym_L13rank320_v1.py)
- [code/fable_oct19_kc_v1/candidate_r13_352_sci_v1.py](code/fable_oct19_kc_v1/candidate_r13_352_sci_v1.py)
- [code/fable_oct19_kc_v1/candidate_r13_448_sci_v1.py](code/fable_oct19_kc_v1/candidate_r13_448_sci_v1.py)
- [code/fable_oct19_kc_v1/candidate_r13_512_sci_v1.py](code/fable_oct19_kc_v1/candidate_r13_512_sci_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5891](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5891)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5930](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5930)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5941](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5941)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5942](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5942)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5943](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5943)

<details>
<summary>All result files and records (2)</summary>

- [science7_v1.log](science7_v1.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-join-rank-layer-13-held-384-ranks-320-352) · [Copy provenance](../../../PROVENANCE.json)

