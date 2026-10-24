# Raising the response-row floor saves arithmetic but worsens adjusted error

2026-09-23T01:53Z · experiment · rejected · measured

Raising the response-row selection floor cuts the read bill but the raw error climbs faster at every setting, another cliff.

Recorded measurement:

    floor .01 util .18828 / 1.648e-8; .02 -> util -5.6%/-2.4% (r35/r100), MSE +9.2%/+4.1% (adj +3.1%/+1.6%); .03 -> adj +20%/+26%; .05 -> +86%/+107%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_refit_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/resp_floor_test.py](code/resp_floor_test.py)

## Evidence

- [NOTES.md#L186](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L186)

<details>
<summary>All result files and records (2)</summary>

- [resp_floor_test.log](resp_floor_test.log) · log
- [NOTES.md](../../third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-raising-response-row-floor-saves-arithmetic-but-worsens-adjusted) · [Copy provenance](../../../PROVENANCE.json)

