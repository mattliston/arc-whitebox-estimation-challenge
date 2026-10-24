# Per-unit third cumulants at layers 8-14 are 7-14% low with correlation 0.96-0.97

2026-10-19T22:16:00Z · finding · informational · measured

The card's per-unit pre-activation third cumulant (the diagonal of the K21 plane) at layers 8-14 tracks Monte Carlo truth with correlation 0.96-0.97 but is 7-14% low in regression scale with a structured error that z, z^2 and the skew proxy explain 55-70% of leave-one-network-out; the k3 mean term is about five times the k4 term at these layers. At the terminal layer the k3 error is 46-48% rms with a missing skew-mean coupling (corr(error, z) = -0.8) and a truth-optimal scale of 1.42-1.53. The final-layer per-unit mean error itself has no cheap structure (regressions on z powers make the cross-half error worse), and the error accumulates late.

Recorded measurement:

    L8-14: corr 0.96-0.97, truth-optimal scale 1.07-1.14, rms error 26-33% of rms truth; model a + v^1.5 (b z + c z^2 + d k3n^2) removes 55-70% LONO (rms 0.27-0.33 -> 0.16-0.18) on 3 fresh nets; L15: rms error 46-48% (5 fresh nets), corr(err, z) -0.8, truth-optimal scale 1.42-1.53, model removes 72-78% (b = -0.0135); L15 mean-error regression on z, z^2, |z|, z^3 (+ v, k3n, k4n): LONO U worse +0.6..+3%; per-layer cross-half U on 7100000: L4 1.8e-9, L8 5.0e-9, L11 8.6e-9, L13 1.3e-8, L14 1.4e-8, L15 1.6e-8

## Programs

- [../2026-10-19-monte-carlo-cumulant-comparison-variance-rms-error-0-17/mc_cumulants_v1.py](../2026-10-19-monte-carlo-cumulant-comparison-variance-rms-error-0-17/mc_cumulants_v1.py)
- [candidate_k3l15_sci_v1.py](candidate_k3l15_sci_v1.py)
- [candidate_k3shape_f_sci_v1.py](candidate_k3shape_f_sci_v1.py)
- [../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py](../../../instruments/fresh-network-truth-generator-matches-organizer-results-after-masking/code/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6043](../../TEAM_LOG_2026-10-01_to_10-22.md#L6043)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6045](../../TEAM_LOG_2026-10-01_to_10-22.md#L6045)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6049](../../TEAM_LOG_2026-10-01_to_10-22.md#L6049)

<details>
<summary>All result files and records (2)</summary>

- [k3l15_coef_v1.json](k3l15_coef_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-19-per-unit-third-cumulants-layers-8-14-7-14) · [Copy provenance](../../../PROVENANCE.json)

