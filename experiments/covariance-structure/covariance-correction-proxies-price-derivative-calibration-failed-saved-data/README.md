# Covariance correction proxies and Price-derivative calibration failed saved-data tests

2026-10-18T04:18:11Z · experiment · rejected · measured

Coherent covariance fitting, positive-semidefinite repair, an eight-direction covariance correction and Price-derivative calibration were screened on saved states. All measured variants lacked an adjusted benefit and were not promoted to native scoring.

Recorded measurement:

    Covariance correction from H, 8 networks: +.718% adjusted → 1.462166e-9; Price-derivative correction: +.082% → 1.452940e-9 versus #336854; fixed-correlation H control variate on 8 networks: +.710% → 1.461805e-9 versus #336867.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_oct18_accuracy_v1/probe_coherent_cov_v1.py](code/astra_oct18_accuracy_v1/probe_coherent_cov_v1.py)
- [code/astra_oct18_accuracy_v1/probe_cov_psd_v1.py](code/astra_oct18_accuracy_v1/probe_cov_psd_v1.py)
- [code/astra_oct18_accuracy_v1/probe_current_H_cov_rank8_v1.py](code/astra_oct18_accuracy_v1/probe_current_H_cov_rank8_v1.py)
- [code/astra_oct18_accuracy_v1/probe_price_covariance_v1.py](code/astra_oct18_accuracy_v1/probe_price_covariance_v1.py)
- [code/astra_oct18_accuracy_v1/probe_rho_H_cov_rank8_v1.py](code/astra_oct18_accuracy_v1/probe_rho_H_cov_rank8_v1.py)
- [code/astra_oct18_bill_v1/partial_mean_jvp_v1.py](code/astra_oct18_bill_v1/partial_mean_jvp_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5171](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5171)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5206](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5206)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5213](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5213)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [COV_PSD_AUDIT_v1.json](COV_PSD_AUDIT_v1.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-18-covariance-correction-proxies-price-derivative-calibration-failed-saved-data) · [Copy provenance](../../../PROVENANCE.json)

