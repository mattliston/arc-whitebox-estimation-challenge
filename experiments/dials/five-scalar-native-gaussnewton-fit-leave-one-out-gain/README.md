# Five-scalar native Gauss–Newton fit: leave-one-out gain is negative

2026-09-28 · experiment · rejected · measured

A fixed five-scalar native Gauss–Newton fit had too little forecast gain and failed leave-one-out checks. No candidate was exported.

Recorded measurement:

    Forecast gain0.035488%;leave-one-out gain-0.136715%,2/6wins.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_native_five_knobs_v1/fit.py](code/hill_climb_native_five_knobs_v1/fit.py)
- [code/hill_climb_native_five_knobs_v1/fit_initial.py](code/hill_climb_native_five_knobs_v1/fit_initial.py)
- [code/hill_climb_native_five_knobs_v1/perturb_edge.py](code/hill_climb_native_five_knobs_v1/perturb_edge.py)
- [code/hill_climb_native_five_knobs_v1/perturb_iso.py](code/hill_climb_native_five_knobs_v1/perturb_iso.py)
- [code/hill_climb_native_five_knobs_v1/perturb_k21n.py](code/hill_climb_native_five_knobs_v1/perturb_k21n.py)
- [code/hill_climb_native_five_knobs_v1/perturb_k31.py](code/hill_climb_native_five_knobs_v1/perturb_k31.py)
- [code/hill_climb_native_five_knobs_v1/perturb_rc.py](code/hill_climb_native_five_knobs_v1/perturb_rc.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)
- [code/native_eval.py](code/native_eval.py)

## Evidence

- [HILL_CLIMB_CURRENT_before_332738.md#L23](../../third-order-planes/common-factor-rank-256-layers-79-16-network-adjusted/HILL_CLIMB_CURRENT_before_332738.md#L23)

<details>
<summary>All result files and records (1)</summary>

- [HILL_CLIMB_CURRENT_before_332738.md](../../third-order-planes/common-factor-rank-256-layers-79-16-network-adjusted/HILL_CLIMB_CURRENT_before_332738.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-five-scalar-native-gaussnewton-fit-leave-one-out-gain) · [Copy provenance](../../../PROVENANCE.json)

