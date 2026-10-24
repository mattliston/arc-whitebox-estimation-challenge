# Refit of four displaced global dials adopted after zero-gradient covariance screens

2026-10-10T13:30Z · decision · adopted · measured

Exact gradients of every per-layer dial family at jmG's point: the new low-rank covariance corrections carry nothing beyond mean-correction covariance-fold coefficient (kappa_C)'s direction, but birth-leg coefficient (X1) (birth-leg), birth scale coefficient (bscale), third-order row-factor coefficient (c0) and pair-map third-cumulant coefficient (k31) are off their optimum after the arrays moved; joint global-dial and learned-array refit (jmH) re-solved them and became candidate generation 45 (S45).

Recorded measurement:

    cvss 1.5, cvgg 1.4, cvgd 0.9 (noise); X1 13 layers beyond 2.5 se, mean t^2 17.1; bscale 6.3; c0 4.6; k31 4.3; jmH: X1 0.557 -> 0.708, c0 0.534 -> 0.517, k31 -0.212 -> +0.107

## Programs

- [jmix15.sh](jmix15.sh)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [candidate.py](candidate.py)
- [../2026-10-10-joint-learned-array-head-input-refit-adopted-2-006/pairs.py](../2026-10-10-joint-learned-array-head-input-refit-adopted-2-006/pairs.py)
- [../2026-10-10-joint-learned-array-head-input-refit-adopted-2-006/jmix.py](../2026-10-10-joint-learned-array-head-input-refit-adopted-2-006/jmix.py)
- [../../../instruments/native-parameter-export-supports-ten-fitted-dials-gradient-direction/code/opus_reexam_20261001_v1/trainer/export.py](../../../instruments/native-parameter-export-supports-ten-fitted-dials-gradient-direction/code/opus_reexam_20261001_v1/trainer/export.py)
- [../2026-10-10-joint-learned-array-head-input-refit-adopted-2-006/export_dials.py](../2026-10-10-joint-learned-array-head-input-refit-adopted-2-006/export_dials.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L636](../../TEAM_LOG_2026-10-01_to_10-22.md#L636)
- [TEAM_LOG_2026-10-01_to_10-22.md#L637](../../TEAM_LOG_2026-10-01_to_10-22.md#L637)
- [TEAM_LOG_2026-10-01_to_10-22.md#L640](../../TEAM_LOG_2026-10-01_to_10-22.md#L640)
- [TEAM_LOG_2026-10-01_to_10-22.md#L643](../../TEAM_LOG_2026-10-01_to_10-22.md#L643)
- [TEAM_LOG_2026-10-01_to_10-22.md#L645](../../TEAM_LOG_2026-10-01_to_10-22.md#L645)
- [TEAM_LOG_2026-10-01_to_10-22.md#L647](../../TEAM_LOG_2026-10-01_to_10-22.md#L647)
- [TEAM_LOG_2026-10-01_to_10-22.md#L644](../../TEAM_LOG_2026-10-01_to_10-22.md#L644)

<details>
<summary>All result files and records (3)</summary>

- [physics_dials.json](physics_dials.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [dial_table_0cvg.out](dial_table_0cvg.out) · result

</details>

## Related work

- enables: [Global dials near optimum after refit; remaining late-band signals require per-layer constants](../../../experiments/dials/global-dials-near-optimum-after-refit-remaining-late-band/README.md)
- shipped in: [Submission #335213: refitted learned arrays and jointly re-solved estimator dials](../../../submissions/335213/README.md)
- ← enables: [Refit protocol separates arm sizing and acceptance onto independent fresh sets of 16 networks](../2026-10-10-refit-protocol-separates-arm-sizing-acceptance-onto-independent-fresh/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-refit-four-displaced-global-dials-adopted-after-zero-gradient) · [Copy provenance](../../../PROVENANCE.json)

