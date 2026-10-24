# Gate-readout learning curve: held-out change reaches -0.826% with 72 training networks

2026-10-21T18:24:37Z · experiment · informational · measured

The fixed held-out fold was measured at four training-set sizes. The recorded extrapolation suggested too little hosted improvement to justify adding the protected 25-network validation set to training.

Recorded measurement:

    - 2026-10-21T18:24:37Z [fable-lead] LEARNING CURVE (lc_gate_fold0_v1, gate 705-col fit, fold0 held-18): n_train 18 +0.076%, 36 -0.502%, 54 -0.683%, 72 -0.826% (18/18 lower). 1/n asymptote ~-1.15%; projected -0.95% at 115 nets -> <= +0.05% hosted: held25 capture NOT worth a shot (teammate redirected to cheaper wall-margin levers: size-cap thresholds, slow-10 walls). Ridge .03 stacks VM32: s15_cap3 all32 -0.253% held25 -0.450% raw -0.615% bill +0.364%, fresh12 -0.461% (8/12); s20_cap3 all32 -0.036% held25 -0.299%, fresh12 -0.222% (7/12). Next: Mac32 r003_s15_cap3; build r003_s15_cap4 (endgame gun).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_bill_v1/on_policy_first128_head90_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lc_gate_fold0_v1.py](code/lc_gate_fold0_v1.py)
- [code/build_parent_sources_v1.py](code/build_parent_sources_v1.py)
- [code/fit_extended_gate_v1.py](code/fit_extended_gate_v1.py)
- [code/fit_second_step_v1.py](code/fit_second_step_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L7523](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L7523)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-gate-readout-learning-curve-held-out-change-reaches-0) · [Copy provenance](../../../PROVENANCE.json)

