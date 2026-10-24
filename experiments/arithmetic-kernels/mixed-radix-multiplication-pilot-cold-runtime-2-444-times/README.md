# Mixed-radix multiplication pilot: cold runtime is 2.444 times the reference

2026-09-27 · experiment · rejected · measured

The fixed mixed-radix runtime pilot failed its speed gate at the first fixture. A static zero-cost accounting bug was reconciled offline without rerunning the already failed runtime experiment.

Recorded measurement:

    Cold runtime2.444x versus1.5x gate;new ternary global saving forecast0.11/0.25%;768 depth-seven forecast0.065/0%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v1/benchmark.py](code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v1/benchmark.py)
- [code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v1/kernel.py](code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v1/kernel.py)
- [code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v1/static_meter.py](code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v1/static_meter.py)
- [code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v2/benchmark.py](code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v2/benchmark.py)
- [code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v2/kernel.py](code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v2/kernel.py)
- [code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v2/static_meter.py](code/codex_resume_20260926_v2/teammate_mixedradix_pilot_v2/static_meter.py)
- [code/astra_phase2_reexamination_v1/smirnov_stream8_native_v1.py](code/astra_phase2_reexamination_v1/smirnov_stream8_native_v1.py)
- [code/astra_phase2_reexamination_v1/streamed_leaf_winograd_v1.py](code/astra_phase2_reexamination_v1/streamed_leaf_winograd_v1.py)
- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py)

## Evidence

- [NEXT_RESEARCH_v1.md#L16](../../dials/covariance-convention-correction-strength-three-short-screens-stay-below/NEXT_RESEARCH_v1.md#L16)

<details>
<summary>All result files and records (1)</summary>

- [NEXT_RESEARCH_v1.md](../../dials/covariance-convention-correction-strength-three-short-screens-stay-below/NEXT_RESEARCH_v1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-mixed-radix-multiplication-pilot-cold-runtime-2-444-times) · [Copy provenance](../../../PROVENANCE.json)

