# Kink-unit fraction falls from 0.490 at layer 8 to 0.348 at layer 14 across 104 generated networks

2026-10-13T01:08Z · finding · informational · measured

For the feature-gated beta the population mean and sd of the kink fraction f_kink(l) (share of all 1024 units with |alpha| < 1.5, alpha = mean / sd of the pre-activation from exact truths) were tabulated per layer over candidate generation 51 (S51)'s 104 generated networks excluding the test networks; the moment-closure estimator's own alpha gives offsets of +0.002..+0.005 at layers 8-14 (0.08-0.18 sd), so the gate used its own definition's table. The population mean and sd of lambda_2 / lambda_1 of the pre-activation covariance over the 64 A36 generated networks were tabulated the same way.

Recorded measurement:

    f_kink L8-14 mu 0.490 / 0.458 / 0.430 / 0.405 / 0.386 / 0.365 / 0.348, sd 0.029 / 0.028 / 0.027 / 0.029 / 0.026 / 0.027 / 0.026; card-alpha offset +0.002..+0.005 at L8-14 = 0.08-0.18 sd; lambda_2 / lambda_1 of Cpre L8-14 mu 0.651 / 0.610 / 0.560 / 0.531 / 0.501 / 0.467 / 0.442, sd 0.067 / 0.065 / 0.066 / 0.069 / 0.070 / 0.066 / 0.059 (64 A36 networks 7100264-327, 16,384-input MC)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1851](../../TEAM_LOG_2026-10-01_to_10-22.md#L1851)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1859](../../TEAM_LOG_2026-10-01_to_10-22.md#L1859)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1892](../../TEAM_LOG_2026-10-01_to_10-22.md#L1892)

<details>
<summary>All result files and records (5)</summary>

- [gbg_mu_sd_v1.json](gbg_mu_sd_v1.json) · result
- [gbg_mu_sd_v1c.json](gbg_mu_sd_v1c.json) · result
- [step0_fkink_v2.json](step0_fkink_v2.json) · result
- [gbg_eig_mu_sd_v1.json](gbg_eig_mu_sd_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-kink-unit-fraction-falls-0-490-layer-8-0) · [Copy provenance](../../../PROVENANCE.json)

