# Matched terminal-readout refits fail to recover the row-sum and diagonal kurtosis correction

2026-10-01T01:22Z · experiment · rejected · measured

Fresh native features from 20 networks in each of two matched physical models feed fixed-ridge five-fold fits of the 65-feature terminal readout. Both the corrected model and the fitted parent worsen every fold against the unchanged parent, and the corrected model also loses against the equally fitted control. No fitted readout is exported; this result does not test a full propagation or hidden-network refit.

Recorded measurement:

    ridge 0.001; 40 captures plus two controls; corrected+fit raw +0.517634796%, adjusted +0.719536257%, 0/5 folds; parent+fit raw +0.402625029%, adjusted +0.405765680%, 0/5; corrected+fit versus fitted parent adjusted +0.312502549%, 2/5

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_rrpi_k4_rowdiag_coadapt_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/common.py](code/common.py)
- [code/collect.py](code/collect.py)
- [code/fit.py](code/fit.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L167](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L167)
- [HILL_CLIMB_CURRENT.md#L172](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L172)
- [cv_result.json#L17](cv_result.json#L17)
- [closeout.md#L1](closeout.md#L1)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [protocol.json](protocol.json) · record
- [cv_result.json](cv_result.json) · result
- [closeout.md](closeout.md) · record
- [captures_complete.json](captures_complete.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-matched-terminal-readout-refits-fail-recover-row-sum-diagonal) · [Copy provenance](../../../PROVENANCE.json)

