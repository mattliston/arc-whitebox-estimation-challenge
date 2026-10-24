# Memoryless (2,1) residual-plane removal raises raw error 52% for a 12% arithmetic saving

2026-09-22T21:08Z · experiment · rejected · measured

Dropping the star carrier, which transports the (2,1) plane residual without memory, costs far more raw error than its 12% of the bill is worth, whether removed at all layers or only early or late.

Recorded measurement:

    star_pre zeroed all layers -> 2.505e-8 (+52%); layers >= 8 only -> 2.054e-8 (+25%); layers < 8 only -> 2.099e-8 (+27%); control 1.648e-8

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_oldtier_economics_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/star_ablate.py](code/star_ablate.py)

## Evidence

- [NOTES.md#L56](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L56)

<details>
<summary>All result files and records (3)</summary>

- [star_ablate_r35.log](star_ablate_r35.log) · log
- [star_ablate_candidate.json](star_ablate_candidate.json) · result
- [NOTES.md](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-memoryless-2-1-residual-plane-removal-raises-raw-error) · [Copy provenance](../../../PROVENANCE.json)

