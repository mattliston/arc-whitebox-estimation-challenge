# Removing third-cumulant leakage preserved plane accuracy and left the mean-error hypothesis unsupported

2026-08-30 · experiment · informational · measured

HONEST TABLES (k3u carry fixed, every input genuinely estimated, validation frame): planes unchanged (L12 rel 0.0802 - the carry change costs nothing plane-wise); dv_corr 0.802/0.823; FIXED oracle-tuned betas now dead (+0.6%/-2.0%) but ESTIMATE-QUALITY beta transfer survives honestly: fit35->apply100 13.6% (2.374e-8), fit100->apply35 10.7% (2.390e-8) - the true deployable-class state: ~12% held-out, 2.37-2.39e-8. Fitted weights: v ~0.44, k3 0.13-0.19, k4 ~0 (k4u supply does not convert). ALSO REFUTED in one probe: the mean-error subtraction hypothesis for the k4xm mode (predicted err = -v x dmu - 2C dmu with coef exactly -1; measured capture 0-3% with unstable coefs vs k4xm's 28-58%).

Recorded measurement:

    HONEST TABLES (k3u carry fixed, every input genuinely estimated, validation frame): planes unchanged (L12 rel 0.0802 - the carry change costs nothing plane-wise); dv_corr 0.802/0.823; FIXED oracle-tuned betas now dead (+0.6%/-2.0%) but ESTIMATE-QUALITY beta transfer survives honestly: fit35->apply100 13.6% (2.374e-8), fit100->apply35 10.7% (2.390e-8) - the true deployable-class state: ~12% held-out, 2.37-2.39e-8. Fitted weights: v ~0.44, k3 0.13-0.19, k4 ~0 (k4u supply does not convert). ALSO REFUTED in one probe: the mean-error subtraction hypothesis for the k4xm mode (predicted err = -v x dmu - 2C dmu with coef exactly -1; measured capture 0-3% with unstable coefs vs k4xm's 28-58%).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g19_attribution_v1/build_gate1_supplies.py](code/public_g19_attribution_v1/build_gate1_supplies.py)
- [code/public_g19_attribution_v1/run_attribution.py](code/public_g19_attribution_v1/run_attribution.py)
- [code/public_g19_attribution_v1/run_beta_transfer.py](code/public_g19_attribution_v1/run_beta_transfer.py)
- [code/public_g19_attribution_v1/run_estimated_consumption.py](code/public_g19_attribution_v1/run_estimated_consumption.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [consumption_hon.log#L1](consumption_hon.log#L1)
- [beta_transfer_hon.log#L1](beta_transfer_hon.log#L1)

<details>
<summary>All result files and records (2)</summary>

- [consumption_hon.log](consumption_hon.log) · log
- [beta_transfer_hon.log](beta_transfer_hon.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-30-removing-third-cumulant-leakage-preserved-plane-accuracy-left-mean) · [Copy provenance](../../../PROVENANCE.json)

