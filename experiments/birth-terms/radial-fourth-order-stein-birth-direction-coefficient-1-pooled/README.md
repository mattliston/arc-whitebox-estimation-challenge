# Radial fourth-order Stein birth direction at coefficient 1: pooled raw ratio 1.000918717, rejected

2026-09-19 · experiment · rejected · measured

A fifth birth-base direction F5_ij = -C_ij * excess_i * (alpha_i + alpha_i^3) * phi_i / 24, the first-order fourth-cumulant contribution to Cov(ReLU(mu_i+X_i), X_j) under the card's radial fourth law K31_ij = excess_i v_i C_ij, was added to the I95 card at coefficient 1 with its diagonal zeroed and column projection on the base removed. An independent Gaussian H4 integral confirmed the sign and coefficient, but both development nets worsened slightly, so the unrefitted field was closed without tuning.

Recorded measurement:

    integral check maximum discrepancy 5.20e-17 at seven alpha values; added cost 220.508 MF on rank 35 (about .05% of I95); native35/100 final raw ratios 1.000498941 / 1.001345955, pooled 1.000918717; prediction delta MSE about 2.1e-12

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/fourth_birth_I95.py](code/codex_resume_20260919_v1/fourth_birth_I95.py)
- [code/codex_resume_20260919_v1/build_fourth_birth.py](code/codex_resume_20260919_v1/build_fourth_birth.py)
- [code/codex_resume_20260919_v1/native_screen.py](code/codex_resume_20260919_v1/native_screen.py)
- [code/fable_wall_repair_v1/cand_c2_wallrepair_I95.py](code/fable_wall_repair_v1/cand_c2_wallrepair_I95.py)

## Evidence

- [RECEIPTS.md#L147](../birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L147)
- [RECEIPTS.md#L150](../birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L150)
- [FOURTH_BIRTH_SCREEN.md#L14](FOURTH_BIRTH_SCREEN.md#L14)
- [FOURTH_BIRTH_SCREEN.md#L17](FOURTH_BIRTH_SCREEN.md#L17)

<details>
<summary>All result files and records (5)</summary>

- [FOURTH_BIRTH_SCREEN.md](FOURTH_BIRTH_SCREEN.md) · record
- [fourth_birth_35_100_score.json](fourth_birth_35_100_score.json) · result
- [fourth_birth_integral_check.json](fourth_birth_integral_check.json) · result
- [fourth_birth_build.json](fourth_birth_build.json) · config
- [RECEIPTS.md](../birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-radial-fourth-order-stein-birth-direction-coefficient-1-pooled) · [Copy provenance](../../../PROVENANCE.json)

