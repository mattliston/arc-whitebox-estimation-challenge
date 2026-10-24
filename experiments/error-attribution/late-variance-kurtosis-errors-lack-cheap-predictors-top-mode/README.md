# Late variance and kurtosis errors lack cheap predictors; a top-mode variance dial adds about 130% MSE

2026-10-13T00:42Z · experiment · rejected · measured

The moment-closure estimator's k4 is one constant per layer while the truth has per-unit sd 0.014-0.017 around 0.06; per-unit features explain 11% of the weighted k4 error, transport sums ~0 (the kurtosis is collective); the L15 variance error has no common mode beyond a weak sign; a dial along the top covariance mode sits at the bottom of a steep bowl.

Recorded measurement:

    k4 features R^2 0.27 in sample; v_15 mode shares R^2 0.08-0.10, ~0 across networks; T132 d = +0.02 +130.3%, -0.02 +124.2% (0 of 16), best d -0.0002 = -0.018%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/physicist/t128_kurt_modes.py](code/opus_reexam_20261001_v1/physicist/t128_kurt_modes.py)
- [code/opus_reexam_20261001_v1/physicist/t128_v_modes.py](code/opus_reexam_20261001_v1/physicist/t128_v_modes.py)
- [code/opus_reexam_20261001_v1/physicist/t131_table.py](code/opus_reexam_20261001_v1/physicist/t131_table.py)
- [code/opus_reexam_20261001_v1/physicist/t114_k4_probe.py](code/opus_reexam_20261001_v1/physicist/t114_k4_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/trainer/otr_common.py](code/opus_reexam_20261001_v1/trainer/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1161](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1161)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1532](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1532)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1534](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1534)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1540](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1540)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1595](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1595)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1833](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1833)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Truth variance at layers 12-15: -24.6% final MSE; third and fourth cumulants give -4.2% and -4.8%](../../../records/research/2026-10-10-truth-variance-layers-12-15-24-6-final-mse/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-late-variance-kurtosis-errors-lack-cheap-predictors-top-mode) · [Copy provenance](../../../PROVENANCE.json)

