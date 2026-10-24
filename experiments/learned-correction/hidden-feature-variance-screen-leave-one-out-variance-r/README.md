# Hidden-feature variance screen: leave-one-out variance R^2 .01094 but mean error +1.06234%

2026-10-21T03:04:54Z · experiment · rejected · measured

The fixed hidden-feature variance screen did not reach the stated variance-prediction gate and worsened mean innovation on all four leave-one-out networks. No source was exported.

Recorded measurement:

    V current-hidden variance screen CLOSED: TRAINvarianceR2 .0330,LOO .01094 (3/4positive), but meaninnovation +1.06234%/0of4lower. No source/export/native. Firstattempt failed pdf reconstruction assertion beforefit; v2 preserved2e-6reconstruction tolerance/nonnegativeclamp only, guard95400PASS414MB/7.5CPU seconds. Perfect-varianceTRAIN diagnostic is a ceiling, not feasiblegain. V now seeks direct covariance information and cheaper recovery of prior expensive accuracycomponents.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_vm_frontier_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fit_current_msg8_messages_v2.py](code/fit_current_msg8_messages_v2.py)
- [code/fit_frontier_v1.py](code/fit_frontier_v1.py)
- [code/fit_hidden_complement_innovation_v1.py](code/fit_hidden_complement_innovation_v1.py)
- [code/hidden_variance_readout_v1/screen_v1.py](code/hidden_variance_readout_v1/screen_v1.py)
- [code/hidden_variance_readout_v1/screen_v2.py](code/hidden_variance_readout_v1/screen_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6489](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6489)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-hidden-feature-variance-screen-leave-one-out-variance-r) · [Copy provenance](../../../PROVENANCE.json)

