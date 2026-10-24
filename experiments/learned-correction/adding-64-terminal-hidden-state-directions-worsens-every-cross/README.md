# Adding 64 terminal hidden-state directions worsens every cross-validation fold

2026-10-01T04:36Z · experiment · rejected · measured

The 128-dimensional terminal hidden activation supplies 64 new readout directions in the kernel of the existing 64-dimensional encoding, while the prior readout and propagation stay fixed. A fixed ridge-0.001 fit on 20 captured networks reproduces the baseline exactly before fitting but worsens all five network folds. The experiment exports no endpoint.

Recorded measurement:

    raw +0.502153616992%; adjusted-cost proxy +0.503915431244%; 0/5 raw and adjusted fold wins; added-dimension rank 64; exact float32 replay; work proxy parent+300000 operations; 20 captures in 514.881799 seconds, peak memory 5.713445 GB, zero added swap

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_rrpi_terminal128_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/collect.py](code/collect.py)
- [code/fit.py](code/fit.py)
- [code/prepare_capture.py](code/prepare_capture.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L36](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L36)
- [HILL_CLIMB_CURRENT.md#L37](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L37)
- [HILL_CLIMB_CURRENT.md#L44](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L44)
- [HILL_CLIMB_CURRENT.md#L54](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L54)
- [cv_result.json#L29](cv_result.json#L29)
- [closeout.json#L19](closeout.json#L19)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [protocol.json](protocol.json) · record
- [cv_result.json](cv_result.json) · result
- [closeout.json](closeout.json) · record
- [features_frozen_before_targets.json](features_frozen_before_targets.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-adding-64-terminal-hidden-state-directions-worsens-every-cross) · [Copy provenance](../../../PROVENANCE.json)

