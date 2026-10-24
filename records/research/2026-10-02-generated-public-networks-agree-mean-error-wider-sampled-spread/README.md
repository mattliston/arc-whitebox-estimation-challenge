# Generated and public networks agree in mean error; wider sampled spread is consistent with small samples

2026-10-02T05:24Z · finding · informational · measured

The base torch card's final-layer error on 33-35 generated seeds has the same mean as on public networks (+0.26, se 0.58) but a 1.4-1.6x wider spread. The truth pipeline was certified on 6 public networks against the organizers' truth (mean difference -0.07e-9, rms 0.86e-9), the layer-15 variance census from the weights alone is the same law (0.0754 vs 0.0756), and the learned correction treats both populations alike (-12.9% vs -12.8% off against on); the proper public reference is the 324 fresh public networks of the trainer's captures (sd 2.28e-9), against which the generated spread ratio 1.4 is reached by 0.4-1.5% of resamples: read as sampling. Final-layer error goes with the layer-15 variance (log-log slope 0.55 on 304 public networks).

Recorded measurement:

    generated 33 seeds 18.68e-9 measured = 17.48e-9 after truth noise 1.20e-9, sd 3.15e-9; public 100 held-out 17.21e-9, sd 1.96e-9 (p 0.001 for spread); 324 fresh public 17.24e-9, sd 2.28e-9; generated 35: 17.64e-9, sd 3.23e-9, ratio 1.4; 6-network certificate: 12.53/13.17, 13.08/12.80, 14.04/13.28, 17.00/17.82, 22.06/20.54, 22.35/23.00 e-9, rms 0.86e-9 (modelled noise 0.61e-9); variance census 0.0754 (sd 12.3%, 140 public) vs 0.0756 (sd 13.1%, 300 generated); S37 vs S36 per board network -4.28% [-5.17,-3.38], 45 of 50

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L117](../../TEAM_LOG_2026-10-01_to_10-22.md#L117)
- [TEAM_LOG_2026-10-01_to_10-22.md#L124](../../TEAM_LOG_2026-10-01_to_10-22.md#L124)
- [TEAM_LOG_2026-10-01_to_10-22.md#L131](../../TEAM_LOG_2026-10-01_to_10-22.md#L131)
- [RESULT.md#L618](../2026-10-01-torch-research-estimator-lowers-final-layer-error-0-73/RESULT.md#L618)
- [RESULT.md#L700](../2026-10-01-torch-research-estimator-lowers-final-layer-error-0-73/RESULT.md#L700)

<details>
<summary>All result files and records (7)</summary>

- [a15_population_check.md](a15_population_check.md) · result
- [a15_reading.md](a15_reading.md) · result
- [a15_pubcert_eval.md](a15_pubcert_eval.md) · result
- [a15_off_report.md](a15_off_report.md) · result
- [a15_public460_relation.md](a15_public460_relation.md) · result
- [RESULT.md](../2026-10-01-torch-research-estimator-lowers-final-layer-error-0-73/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-02-generated-public-networks-agree-mean-error-wider-sampled-spread) · [Copy provenance](../../../PROVENANCE.json)

