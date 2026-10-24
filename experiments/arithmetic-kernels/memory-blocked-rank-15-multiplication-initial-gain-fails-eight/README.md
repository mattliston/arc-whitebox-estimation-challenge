# Memory-blocked rank-15 multiplication: initial gain fails eight-network expansion

2026-09-28 · experiment · rejected · measured

The blocked implementation preserved each multiplication and original arena size, with exact first-network parity to the failed-memory integration. Native2 passed, but the fixed eight-network gain, win and raw-error gates failed.

Recorded measurement:

    Native2 adjusted gain1.596807568%,2/2wins,raw+.132522538%;TRAIN8 adjusted-.47415648380658704%,raw+.35514332969242%,bill-.8489700965794489%,5/8wins,CI[-.9991990484410722%,-.05484216617511207%].

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_hk15_blocked_v2/candidate.py](code/hill_climb_hk15_blocked_v2/candidate.py)
- [code/hill_climb_exact_bundle_v1/candidate.py](code/hill_climb_exact_bundle_v1/candidate.py)
- [code/hill_climb_hk15_blocked_v2/assess_native.py](code/hill_climb_hk15_blocked_v2/assess_native.py)
- [code/hill_climb_hk15_blocked_v2/build_candidate.py](code/hill_climb_hk15_blocked_v2/build_candidate.py)
- [code/hill_climb_hk15_blocked_v2/build_fixed_plan.py](code/hill_climb_hk15_blocked_v2/build_fixed_plan.py)
- [code/hill_climb_hk15_blocked_v2/constants.py](code/hill_climb_hk15_blocked_v2/constants.py)
- [code/hill_climb_hk15_blocked_v2/dispatcher.py](code/hill_climb_hk15_blocked_v2/dispatcher.py)
- [code/hill_climb_hk15_blocked_v2/kernel_blocked.py](code/hill_climb_hk15_blocked_v2/kernel_blocked.py)
- [code/hill_climb_hk15_blocked_v2/native_probe.py](code/hill_climb_hk15_blocked_v2/native_probe.py)
- [code/hill_climb_hk15_blocked_v2/setup.py](code/hill_climb_hk15_blocked_v2/setup.py)
- [code/hill_climb_hk15_blocked_v2/shared_fixtures.py](code/hill_climb_hk15_blocked_v2/shared_fixtures.py)
- [code/hill_climb_hk15_integration_v1/kernel_borrowed.py](code/hill_climb_hk15_integration_v1/kernel_borrowed.py)

## Evidence

- [HILL_CLIMB_before_g4_symdiag_boundary_v1.md#L33](../../third-order-planes/canonical-polyadic-tensor-star-consistency-adjusted-gain-0-0123402/HILL_CLIMB_before_g4_symdiag_boundary_v1.md#L33)

<details>
<summary>All result files and records (6)</summary>

- [HILL_CLIMB_before_g4_symdiag_boundary_v1.md](../../third-order-planes/canonical-polyadic-tensor-star-consistency-adjusted-gain-0-0123402/HILL_CLIMB_before_g4_symdiag_boundary_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [native2_assessment.json](native2_assessment.json) · result
- [native8_assessment.json](native8_assessment.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-memory-blocked-rank-15-multiplication-initial-gain-fails-eight) · [Copy provenance](../../../PROVENANCE.json)

