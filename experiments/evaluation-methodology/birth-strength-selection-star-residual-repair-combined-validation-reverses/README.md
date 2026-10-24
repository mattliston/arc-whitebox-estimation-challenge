# Birth-strength selection and star residual repair combined: validation reverses training gain

2026-09-28 · experiment · rejected · measured

The birth distinct-strength selector and consumer-star repair were combined with the exact-cost component as a newly frozen conjunction. Training passed but the fixed40-network validation failed, leaving the separate exact-cost component intact.

Recorded measurement:

    TRAIN16 adjusted-.753127190%,13/16wins,CI[-1.335730808%,-.137624259%];VALID40 adjusted+.164462482%,17/40wins,CI[-.261396856%,+.602996926%];raw+.244117154%,bill-.071000214%,CPU-.804473622%,max regression4.435014913%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_bds_csr_exact_aggregate_v1/candidate.py](code/hill_climb_bds_csr_exact_aggregate_v1/candidate.py)
- [code/hill_climb_bds_csr_exact_aggregate_v1/build.py](code/hill_climb_bds_csr_exact_aggregate_v1/build.py)
- [code/compare_native.py](code/compare_native.py)
- [code/hill_climb_bds_csr_exact_assess_v1.py](code/hill_climb_bds_csr_exact_assess_v1.py)
- [code/hill_climb_birth_distinct_strength_v2/candidate.py](code/hill_climb_birth_distinct_strength_v2/candidate.py)
- [code/hill_climb_birth_distinct_strength_v2/fragment.py](code/hill_climb_birth_distinct_strength_v2/fragment.py)
- [code/hill_climb_exact_bundle_v1/candidate.py](code/hill_climb_exact_bundle_v1/candidate.py)
- [code/hill_climb_star_consumer_residual_v2/candidate.py](code/hill_climb_star_consumer_residual_v2/candidate.py)
- [code/hill_climb_star_consumer_residual_v2/fragment.py](code/hill_climb_star_consumer_residual_v2/fragment.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)
- [code/native_eval.py](code/native_eval.py)
- [code/team_bank_validation_s32_v3.py](code/team_bank_validation_s32_v3.py)

## Evidence

- [HILL_CLIMB_before_physical11_v1.md#L13](HILL_CLIMB_before_physical11_v1.md#L13)

<details>
<summary>All result files and records (6)</summary>

- [HILL_CLIMB_before_physical11_v1.md](HILL_CLIMB_before_physical11_v1.md) · record
- [RESULT_train16.md](RESULT_train16.md) · record
- [train16_assessment.json](train16_assessment.json) · result
- [broad_assessment.json](broad_assessment.json) · result
- [build.json](build.json) · result
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-28-birth-strength-selection-star-residual-repair-combined-validation-reverses) · [Copy provenance](../../../PROVENANCE.json)

