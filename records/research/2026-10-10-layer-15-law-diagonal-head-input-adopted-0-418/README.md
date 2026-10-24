# Layer-15 law-diagonal head input adopted: -0.418% pooled MSE on 50 networks

2026-10-10T04:15Z · decision · adopted · measured

The trainer's first table and line search used layer 14's alive mask on layer-15 units and were void; with the kept rows |mu_15| <= 3.5 sqrt(v_15) the physicist's offline Gauss-Newton fit of 128 layer-15 weights held out on the judging ranks and was carried into the candidate generation 44 (S44) refit.

Recorded measurement:

    27 held-out ranks -0.42% (se 0.07, 24 of 27); all 50 pooled 1.4177e-8 -> 1.4118e-8 = -0.418% (se 0.071, 41 of 50); bill +0.077%; native S43 + 17 vs S43 -0.28% (se 0.07, 8 of 10)

## Programs

- [t67b_judge17.py](t67b_judge17.py)
- [../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v2.py](../../../instruments/final-layer-head-column-hook-corrected-smoke-tested-25/code/opus_reexam_20261001_v1/engineer/a42_head/fragment_hcf_v2.py)
- [../../../experiments/learned-correction/joint-third-step-three-fourth-order-input-columns-gives/code/cert/head/solve.py](../../../experiments/learned-correction/joint-third-step-three-fourth-order-input-columns-gives/code/cert/head/solve.py)
- [../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](../../../experiments/moment-propagation/consumer-constrained-rank-32-star-basis-increases-adjusted-error/code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py](../../../experiments/learned-correction/continued-training-lowers-held-out-error-only-0-240/code/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L492](../../TEAM_LOG_2026-10-01_to_10-22.md#L492)
- [TEAM_LOG_2026-10-01_to_10-22.md#L504](../../TEAM_LOG_2026-10-01_to_10-22.md#L504)
- [TEAM_LOG_2026-10-01_to_10-22.md#L522](../../TEAM_LOG_2026-10-01_to_10-22.md#L522)
- [TEAM_LOG_2026-10-01_to_10-22.md#L529](../../TEAM_LOG_2026-10-01_to_10-22.md#L529)
- [TEAM_LOG_2026-10-01_to_10-22.md#L531](../../TEAM_LOG_2026-10-01_to_10-22.md#L531)
- [TEAM_LOG_2026-10-01_to_10-22.md#L544](../../TEAM_LOG_2026-10-01_to_10-22.md#L544)
- [TEAM_LOG_2026-10-01_to_10-22.md#L547](../../TEAM_LOG_2026-10-01_to_10-22.md#L547)
- [TEAM_LOG_2026-10-01_to_10-22.md#L562](../../TEAM_LOG_2026-10-01_to_10-22.md#L562)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Joint learned-array and head-input refit adopted: -2.006% on 50 networks without the old-pair quartic](../2026-10-10-joint-learned-array-head-input-refit-adopted-2-006/README.md)
- shipped in: [Submission #335147: joint correction refit, layer-15 law diagonal and padded covariance cut](../../../submissions/335147/README.md)
- ← enables: [Layer-15 weight refit rejected: +1.69% held-out MSE despite -1.77% in sample](../../../experiments/learned-correction/layer-15-weight-refit-rejected-1-69-held-out/README.md)

[Research graph](../../../graph/index.html#node=2026-10-10-layer-15-law-diagonal-head-input-adopted-0-418) · [Copy provenance](../../../PROVENANCE.json)

