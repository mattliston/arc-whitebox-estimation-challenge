# Network-level five-mode calibration: feature head loses and oracle gain is only 1.548859%

2026-09-27 · experiment · rejected · measured

Ninety-four native feature columns were pooled over1024 neurons to predict five shared response coefficients under four network folds. The fixed linear span had little oracle capacity and the learned calibration failed.

Recorded measurement:

    Feature arm raw+0.111002%,95% interval[-0.005283%,+0.229964%],54/128 wins;intercept control-0.055768%,interval[-0.107402%,-0.003881%],76/128 wins;per-network true-answer least squares removes1.548859% pooled error;ridge1.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/network_coherent_readout_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/assess.py](code/assess.py)
- [code/common.py](code/common.py)
- [code/run.py](code/run.py)
- [code/verify.py](code/verify.py)

## Evidence

- [RESULTS.md#L1350](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1350)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [RESULT.md](RESULT.md) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-network-level-five-mode-calibration-feature-head-loses-oracle) · [Copy provenance](../../../PROVENANCE.json)

