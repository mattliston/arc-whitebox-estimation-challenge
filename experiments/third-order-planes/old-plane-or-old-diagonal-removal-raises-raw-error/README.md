# Old-plane or old-diagonal removal raises raw error 11-fold or 14-fold

2026-09-22T20:52Z · experiment · rejected · measured

Zeroing the old tier's plane read (diagonal and radial intact) or its diagonal read (plane intact) on research network 35 each multiplies the final MSE by an order of magnitude, so neither consumer of the old content can be cut.

Recorded measurement:

    old plane read zeroed -> 1.79e-7 (11x); old diagonal zeroed -> 2.28e-7 (14x); control 1.648e-8

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_oldtier_economics_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/consumer_ablate.py](code/consumer_ablate.py)

## Evidence

- [NOTES.md#L32](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L32)

<details>
<summary>All result files and records (3)</summary>

- [consumer_ablate_r35.log](consumer_ablate_r35.log) · log
- [consumer_ablate_candidate.json](consumer_ablate_candidate.json) · result
- [NOTES.md](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-old-plane-or-old-diagonal-removal-raises-raw-error) · [Copy provenance](../../../PROVENANCE.json)

