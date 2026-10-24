# Layer-14 least-squares redistribution of birth and star contributions fails held-out validation

2026-10-01T03:06Z · experiment · rejected · measured

A fixed full-strength layer-14 least-squares redistribution of the post-birth b contribution and star representation is ported without changing its arithmetic. Batching temporary ownership preserves the original helper’s output and charge, but the four-network improvement reverses on the eight-network check. The candidate is not retained.

Recorded measurement:

    n=4; raw -0.260279994%; adjusted -0.246961835%; charged work +0.014376477%; raw wins 3/4; adjusted wins 3/4; adjusted 95% interval [-0.4420908355930675, -0.032491889275321206]%; held-out n=8; raw +0.008454577%; adjusted +0.028646881%; charged work +0.014142636%; raw wins 4/8; adjusted wins 4/8; adjusted 95% interval [-0.15381555860521923, 0.2267543289088093]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_bcarrier_ls14_v1/fragment.py](code/hill_climb_rrpi_bcarrier_ls14_v1/fragment.py)
- [code/hill_climb_rrpi_bcarrier_ls14_v1/candidate.py](code/hill_climb_rrpi_bcarrier_ls14_v1/candidate.py)
- [code/hill_climb_rrpi_bcarrier_ls14_v1/probe.py](code/hill_climb_rrpi_bcarrier_ls14_v1/probe.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/hill_climb_rrpi_bcarrier_ls14_v1/assess.py](code/hill_climb_rrpi_bcarrier_ls14_v1/assess.py)
- [code/hill_climb_rrpi_bcarrier_ls14_v1/prepare.py](code/hill_climb_rrpi_bcarrier_ls14_v1/prepare.py)
- [code/hill_climb_s34_bcarrier_ls14_full_v1/fragment.py](code/hill_climb_s34_bcarrier_ls14_full_v1/fragment.py)
- [code/hill_climb_uniform_gnn_gain_v1/endpoint.py](code/hill_climb_uniform_gnn_gain_v1/endpoint.py)
- [code/hill_climb_uniform_gnn_gain_v1/fit.py](code/hill_climb_uniform_gnn_gain_v1/fit.py)
- [code/native_screen_engine_v1/screen.py](code/native_screen_engine_v1/screen.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L96](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L96)
- [HILL_CLIMB_CURRENT.md#L100](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L100)
- [HILL_CLIMB_CURRENT.md#L101](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L101)
- [HILL_CLIMB_CURRENT.md#L109](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L109)
- [primitive_result.json#L1](primitive_result.json#L1)
- [fit4_assessment.json#L42](fit4_assessment.json#L42)
- [check8_assessment.json#L50](check8_assessment.json#L50)
- [closeout.json#L1](closeout.json#L1)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [primitive_result.json](primitive_result.json) · result
- [fit4_assessment.json](fit4_assessment.json) · result
- [check8_assessment.json](check8_assessment.json) · result
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-layer-14-least-squares-redistribution-birth-star-contributions-fails) · [Copy provenance](../../../PROVENANCE.json)

