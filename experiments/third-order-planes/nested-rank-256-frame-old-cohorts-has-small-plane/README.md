# Nested rank-256 frame for old cohorts has small plane error but insufficient downstream accuracy

2026-09-22T22:18Z · experiment · rejected · measured

A static proxy shows confining old cohorts to a nested rank-256 sub-frame would cost only 0.17-0.28% plane error (worth ~3-5% of the bill), but a native rank dial on the whole tier shows the closure amplifies plane errors about ten times more than the public eps law: rank 352 (-1.5% util) costs +4.7% / +3.8% raw, so a rank-256 sub-frame would be a wash without a refit.

Recorded measurement:

    subframe age>=9: rank 256 plane rel err 0.17-0.28%, diag 0.10-0.16% (energy 99.97-99.998%); rank 192 0.5-0.9%; rank 128 1.3-2.5%; native whole-tier rank 352 -> util -1.5% / MSE +4.7% (r35), +3.8% (r100); 320 -> util -6% / +9.5%, +14.8%; 288 -> util -7.6% / +21.6%, +24.8%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_oldtier_economics_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/subframe_info_loss.py](code/subframe_info_loss.py)
- [code/rank_dial.py](code/rank_dial.py)

## Evidence

- [NOTES.md#L95](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L95)
- [NOTES.md#L98](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L98)

<details>
<summary>All result files and records (4)</summary>

- [subframe_info_loss_r35.log](subframe_info_loss_r35.log) · log
- [rank_dial_r35_r100.log](rank_dial_r35_r100.log) · log
- [rank_dial_256.log](rank_dial_256.log) · log
- [NOTES.md](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-nested-rank-256-frame-old-cohorts-has-small-plane) · [Copy provenance](../../../PROVENANCE.json)

