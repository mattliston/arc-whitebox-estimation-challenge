# Monte Carlo cumulant comparison: variance RMS error 0.17%, fourth-cumulant correlation only 0.72-0.80

2026-10-19T21:27:00Z · finding · informational · measured

A new instrument (1M-sample Monte Carlo of per-unit pre-activation cumulants, 140 s per network) was compared with the card's per-unit internals on five fresh networks: per-unit variance is exact to 0.17% rms, but the card's per-unit k4 at layers 8-14 correlates only 0.72-0.80 with truth and the per-network truth-optimal scalar k4 scale (1.05-1.08) matches that network's MSE-optimal band move, which is the mechanism of the per-network band optima. The per-unit error correlates -0.3 to -0.5 with the card's own carried skew proxy squared and a stable per-layer fit on it removes 12-26% of the error variance; the row fourth-moment term explains none of it (the structure lives in the pair/plane contractions), and 1M-sample k4 truth carries about 20% per-unit noise.

Recorded measurement:

    fresh 7100000-7100004; v rms error 0.17%; k4 corr 0.72-0.80 at L8-14 (true excess kurtosis 0.044-0.060, unit sd 0.009-0.016; rising 0.010 at L1 to 0.060 at L14); per-net truth-optimal scale 1.05-1.08 on 7100000 vs MSE-optimal band move (f2 -1.23%, f3 -2.09% at x1.05); corr(error, k3n^2) -0.3..-0.5 with k3n = 3 bpre/sqrt(vv), corr(k3 proxy, true k3) 0.94-0.95; fit err ~ b1 vv^2 (k3n^2 - mean), b1 = -3.1..-4.2 at L8-14, 12-26% of error variance; row fourth-moment corr ~0.0; MC k4 noise ~20% per unit (k3 ~5%)

## Programs

- [mc_cumulants_v1.py](mc_cumulants_v1.py)
- [probe_k4dump_v1.py](probe_k4dump_v1.py)
- [k4_truth_compare_v1.py](k4_truth_compare_v1.py)
- [k4shape_fit_v1.py](k4shape_fit_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6033](../../TEAM_LOG_2026-10-01_to_10-22.md#L6033)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6035](../../TEAM_LOG_2026-10-01_to_10-22.md#L6035)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6045](../../TEAM_LOG_2026-10-01_to_10-22.md#L6045)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-monte-carlo-cumulant-comparison-variance-rms-error-0-17) · [Copy provenance](../../../PROVENANCE.json)

