# Zeroing third-moment columns on inactive rows raises raw error 4.4–20.7%

2026-09-23T02:16Z · experiment · rejected · measured

Skipping dead rows in the old reconstruct and young push would save about 5-6% of the bill, but zeroing their third-cumulant rows raises the final MSE because the dead units' small means feed the next-layer mean at the 1e-4 level.

Recorded measurement:

    r35: none 1.648017e-8; old rows only 1.720387e-8 (ratio 1.04391); all 1.837565e-8 (1.11502); r100 +7.1% / +20.7%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/deadrow_test.py](code/deadrow_test.py)

## Evidence

- [NOTES.md#L191](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L191)

<details>
<summary>All result files and records (2)</summary>

- [deadrow_test.log](deadrow_test.log) · log
- [NOTES.md](../removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-zeroing-third-moment-columns-inactive-rows-raises-raw-error) · [Copy provenance](../../../PROVENANCE.json)

