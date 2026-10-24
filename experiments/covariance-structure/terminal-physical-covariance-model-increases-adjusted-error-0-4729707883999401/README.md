# Terminal physical covariance model increases adjusted error 0.4729707883999401%

2026-09-27T18:43:56.298397Z · experiment · rejected · measured

A6177-parameter480-update terminal covariance model recomputed physical star feedback and exact native variance with original host parameters frozen. The implementation correction preserved the initial fidelity failure, and the corrected native candidate still lost the actual two-network score.

Recorded measurement:

    TRAIN2 raw+.7667366003294429%,adjusted+.4729707883999401%,1/2adjusted wins;24guards,peak5.626GB,zero swap;no TRAIN6/VALID40.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_physical_covariance_v1/candidate.py](code/terminal_physical_covariance_v1/candidate.py)
- [code/terminal_physical_covariance_v1/build.py](code/terminal_physical_covariance_v1/build.py)
- [code/terminal_physical_covariance_v1/fit.py](code/terminal_physical_covariance_v1/fit.py)
- [code/terminal_physical_covariance_v1/fit_v2.py](code/terminal_physical_covariance_v1/fit_v2.py)
- [code/terminal_physical_covariance_v1/native_bridge.py](code/terminal_physical_covariance_v1/native_bridge.py)
- [code/candidate_old_both_tail_trace_bounded_v6.py](code/candidate_old_both_tail_trace_bounded_v6.py)
- [code/candidate_thin_star_route_v1.py](code/candidate_thin_star_route_v1.py)
- [code/terminal_physical_covariance_v1/common.py](code/terminal_physical_covariance_v1/common.py)
- [code/terminal_physical_covariance_v1/consumer.py](code/terminal_physical_covariance_v1/consumer.py)
- [code/terminal_physical_covariance_v1/consumer_v2.py](code/terminal_physical_covariance_v1/consumer_v2.py)
- [code/terminal_physical_covariance_v1/disabled.py](code/terminal_physical_covariance_v1/disabled.py)
- [code/terminal_physical_covariance_v1/fragment.py](code/terminal_physical_covariance_v1/fragment.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/terminal_physical_covariance_v1/model.py](code/terminal_physical_covariance_v1/model.py)
- [code/terminal_physical_covariance_v1/qualify_v2.py](code/terminal_physical_covariance_v1/qualify_v2.py)

</details>

## Evidence

- [NEXT_RESEARCH.md#L12](../../arithmetic-kernels/batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md#L12)

<details>
<summary>All result files and records (7)</summary>

- [NEXT_RESEARCH.md](../../arithmetic-kernels/batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md) · record
- [result.json](result.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record
- [fit_result.json](fit_result.json) · result
- [qualification_v2.json](qualification_v2.json) · result
- [continuation_checkpoint_v66.json](continuation_checkpoint_v66.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-terminal-physical-covariance-model-increases-adjusted-error-0-4729707883999401) · [Copy provenance](../../../PROVENANCE.json)

