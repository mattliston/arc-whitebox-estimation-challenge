# Sixteen rank-8 correlation-energy terminal features, five-fold ridge: .9988638603 vs unchanged, 1.0024666616 vs affine

2026-09-19 · experiment · rejected · measured

Sixteen covariance-aware energy features of the terminal message (rank 8, two-pass QR, 399.725 MFLOP per network) were added to the fixed 65-column affine head and tested by five-fold network cross-validation on 20 TRAIN networks of the Cost75 card with sealed out-of-fold predictions. The extension beats the unchanged head by 0.11% but loses to the affine control, failing the predeclared gate of at least 1% below unchanged, below affine and 4/5 fold wins.

Recorded measurement:

    pooled adjusted: unchanged 4.0409079922e-9, affine control 4.0263852265e-9, extended 4.0363169563e-9; extended/unchanged .9988638603, extended/control 1.0024666616; no fold beats both; feature cost 399,725,292 F; quadratic-form certificate error 6.73e-8

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/correlated_message_v1/capture_all.py](code/correlated_message_v1/capture_all.py)
- [code/correlated_message_v1/information.py](code/correlated_message_v1/information.py)
- [code/correlated_message_v1/certificate.py](code/correlated_message_v1/certificate.py)
- [code/correlated_message_v1/capture.py](code/correlated_message_v1/capture.py)
- [code/correlated_message_v1/feature.py](code/correlated_message_v1/feature.py)
- [code/cost_rule75_scalar.py](code/cost_rule75_scalar.py)
- [code/run_guarded.py](code/run_guarded.py)

## Evidence

- [RECEIPTS.md#L276](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L276)
- [RECEIPTS.md#L279](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L279)
- [RECEIPTS.md#L281](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L281)
- [RECEIPTS.md#L283](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L283)

<details>
<summary>All result files and records (3)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · result
- [PROTOCOL.md](PROTOCOL.md) · record
- [RECEIPTS.md](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-sixteen-rank-8-correlation-energy-terminal-features-five-fold) · [Copy provenance](../../../PROVENANCE.json)

