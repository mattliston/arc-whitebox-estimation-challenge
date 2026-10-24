# More aggressive tail cutoffs with thin star routing: all three native screens lose

2026-09-27 · experiment · rejected · measured

Negative-only cutoffs2.0 and1.5 and a two-sided cutoff2.0 were tested with the retained thin-star routing. None won either native case.

Recorded measurement:

    TRAIN35/100 adjusted:negative2.0+0.321058%;negative1.5+0.424647%;both2.0+1.628207%;zero adjusted wins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/incumbent_tail_cut_screens_v1/both20.py](code/codex_resume_20260926_v2/incumbent_tail_cut_screens_v1/both20.py)
- [code/codex_resume_20260926_v2/incumbent_tail_cut_screens_v1/negative15.py](code/codex_resume_20260926_v2/incumbent_tail_cut_screens_v1/negative15.py)
- [code/codex_resume_20260926_v2/incumbent_tail_cut_screens_v1/negative20.py](code/codex_resume_20260926_v2/incumbent_tail_cut_screens_v1/negative20.py)
- [code/codex_resume_20260926_v2/incumbent_tail_cut_screens_v1/run.py](code/codex_resume_20260926_v2/incumbent_tail_cut_screens_v1/run.py)
- [code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py](code/codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py)
- [code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py](code/codex_resume_20260926_v2/candidate_old_both_tail_trace_bounded_v6.py)
- [code/codex_resume_20260926_v2/candidate_thin_star_route_v1.py](code/codex_resume_20260926_v2/candidate_thin_star_route_v1.py)
- [code/codex_resume_20260926_v2/compare_native.py](code/codex_resume_20260926_v2/compare_native.py)
- [code/codex_resume_20260926_v2/native_eval.py](code/codex_resume_20260926_v2/native_eval.py)

## Evidence

- [RESULTS.md#L1641](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L1641)

<details>
<summary>All result files and records (1)</summary>

- [RESULTS.md](../all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-27-more-aggressive-tail-cutoffs-thin-star-routing-all-three) · [Copy provenance](../../../PROVENANCE.json)

