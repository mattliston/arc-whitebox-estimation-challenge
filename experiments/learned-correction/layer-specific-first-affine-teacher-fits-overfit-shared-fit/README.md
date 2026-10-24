# Layer-specific first-affine teacher fits overfit; shared fit passes a prespecified fidelity gate

2026-10-21T01:24:59Z · experiment · rejected · measured

Two layer-specific fits worsened teacher validation head error to 201% / 255% of baseline and were rejected before native execution. A shared fit across 15 layers passed its fixed fidelity gate; this was not yet a score gain.

Recorded measurement:

    Two layer-specific 10k-parameter bank8 first-affine fits overfit and were rejected before native: original16 inputs valhead201% of baseline; actual777a inputs valhead255%. Shared actual777a first-affine across15layers/7680trainrows completed guard54915: validation state -4.12%, message -6.96%, head +0.80% versus unmodified inputs, passes predeclared <=5% degradation gate. Fidelity is not adjusted gain. V exports/checks then native32; additionally prepares actual777a state64 head-only anchored ridge (65coefficients/layer), same train4/val3, no truth.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct20_vm_frontier_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/export_bank8_shared_firstaffine_v1.py](code/export_bank8_shared_firstaffine_v1.py)
- [code/fit_bank8_actual_input_firstaffine_v1.py](code/fit_bank8_actual_input_firstaffine_v1.py)
- [code/fit_bank8_actual_input_shared_v1.py](code/fit_bank8_actual_input_shared_v1.py)
- [code/fit_bank8_parent16_firstaffine_v1.py](code/fit_bank8_parent16_firstaffine_v1.py)
- [code/fit_current_msg8_messages_v2.py](code/fit_current_msg8_messages_v2.py)
- [code/fit_frontier_v1.py](code/fit_frontier_v1.py)
- [code/shared_firstaffine_math_v1.py](code/shared_firstaffine_math_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6438](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6438)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-21-layer-specific-first-affine-teacher-fits-overfit-shared-fit) · [Copy provenance](../../../PROVENANCE.json)

