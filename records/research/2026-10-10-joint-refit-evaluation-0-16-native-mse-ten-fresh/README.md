# Joint-refit evaluation: -0.16% native MSE on ten fresh networks versus -0.60% research MSE on 50 ranks

2026-10-10T06:46Z · finding · informational · measured

The candidate generation 43 (S43) joint solve (learned-array step + skew-corrected birth (bsk) 0.34 + cohort-age (3,1) plane term (k31l) age 3 at 0.64) judged -0.60% on 50 ranks in PyTorch but -0.16% natively on ten fresh networks; the two track each other network by network with native weaker by ~0.28%, of which the dead cut explains +0.06%; no build error.

Recorded measurement:

    torch 50 ranks 1.4093e-8 vs 1.4177e-8 = -0.596% (se 0.121, 42 of 50); native ten -0.16% pooled (se 0.26, 6 of 10); torch on the same ten -0.42%

## Programs

- [mix.py](mix.py)
- [candidate.py](candidate.py)
- [parity_level.py](parity_level.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L540](../../TEAM_LOG_2026-10-01_to_10-22.md#L540)
- [TEAM_LOG_2026-10-01_to_10-22.md#L543](../../TEAM_LOG_2026-10-01_to_10-22.md#L543)
- [TEAM_LOG_2026-10-01_to_10-22.md#L562](../../TEAM_LOG_2026-10-01_to_10-22.md#L562)
- [TEAM_LOG_2026-10-01_to_10-22.md#L563](../../TEAM_LOG_2026-10-01_to_10-22.md#L563)
- [TEAM_LOG_2026-10-01_to_10-22.md#L569](../../TEAM_LOG_2026-10-01_to_10-22.md#L569)
- [TEAM_LOG_2026-10-01_to_10-22.md#L583](../../TEAM_LOG_2026-10-01_to_10-22.md#L583)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Joint learned-array and head-input refit adopted: -2.006% on 50 networks without the old-pair quartic](../2026-10-10-joint-learned-array-head-input-refit-adopted-2-006/README.md)
- enables: [Three native approximations explain 91% of the native-versus-research error gap](../2026-10-10-three-native-approximations-explain-91-native-versus-research-error/README.md)
- ← enables: [Request reduction and skew-corrected birth adopted as the next joint-refit base: -0.28% on ten fresh networks](../2026-10-10-request-reduction-skew-corrected-birth-adopted-next-joint-refit/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-joint-refit-evaluation-0-16-native-mse-ten-fresh) · [Copy provenance](../../../PROVENANCE.json)

