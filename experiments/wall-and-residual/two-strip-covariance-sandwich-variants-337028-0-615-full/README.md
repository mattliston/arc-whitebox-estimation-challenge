# Two-strip covariance sandwich variants of #337028: +0.615% full or +0.142% late-only adjusted

2026-10-19T03:12:38Z · experiment · rejected · measured

Two request-diet twins of the #337028 bundle (the two-strip covariance sandwich fragment appended last, over all layers or over layers 6-15 only) were judged on seven ranks and taken through the local gates as candidates for the second designated private-run file: they lower the per-network request count by 7.2% / 4.4% and move the hosted residual into the class that had never lost a network. Both cost adjusted score and were never fired; the owner then ruled out spending effort on safety variants.

Recorded measurement:

    full twin da5f8a61 (7 ranks vs 95c): raw +0.082% (se .17, 2/7 lower), bill +0.533%, adjusted +0.615% (se .16) = +0.86% vs #337028, projected ~1.4560e-9; requests -3,131/net (-7.2%: 44,279 -> 41,027 at rank 35), residual x0.930, flops x1.0067; gates setup5 3.283 s, stage A archive 62a6088929ac (703,993 bytes); partial twin e3e78002 (strips at L6-15): raw -0.117% (se .17, 3/7 lower), bill +0.259%, adjusted +0.142% (se .17) = +0.39% vs #337028, projected ~1.449e-9, requests -1,926/net (-4.4%), residual x0.950 -> hosted median ~.331

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_safe_twin_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_jsym_twostrip_v1.py](code/candidate_jsym_twostrip_v1.py)
- [code/candidate_jsym_twostrip_L6_15_v2.py](code/candidate_jsym_twostrip_L6_15_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5896](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5896)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5897](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5897)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5898](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5898)

<details>
<summary>All result files and records (4)</summary>

- [science7_v1.log](science7_v1.log) · log
- [science7_L6_15_v2.log](science7_L6_15_v2.log) · log
- [release_min_v1.log](release_min_v1.log) · log
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-two-strip-covariance-sandwich-variants-337028-0-615-full) · [Copy provenance](../../../PROVENANCE.json)

