# Variance correction scale reduced from 0.82 to 0.40 and transferred across three networks

2026-09-02 · experiment · informational · measured

The consumption betas were leave-one-out (LOO)-fit for an older supply; on the deploy-frame supply beta_v ~ 0.40 (r35 0.40, r100 0.435; frozen 0.40 sign-stable on 35/100/150). Per-window betas <= +0.5 (not a lever). k3/k4 betas unstable across nets -> v-only.

Code availability: The consumption and cross-network beta-transfer programs were recovered. The original launch configuration for every sweep represented by the two aggregate logs was not identified.

Recorded measurement:

    The consumption betas were LOO-fit for an older supply; on the deploy-frame supply beta_v ~ 0.40 (r35 0.40, r100 0.435; frozen 0.40 sign-stable on 35/100/150). Per-window betas <= +0.5 (not a lever). k3/k4 betas unstable across nets -> v-only.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g19_attribution_v1/build_gate1_supplies.py](code/public_g19_attribution_v1/build_gate1_supplies.py)
- [code/public_g19_attribution_v1/consumption_betas.py](code/public_g19_attribution_v1/consumption_betas.py)
- [code/public_g19_attribution_v1/run_attribution.py](code/public_g19_attribution_v1/run_attribution.py)
- [code/public_g19_attribution_v1/run_beta_transfer.py](code/public_g19_attribution_v1/run_beta_transfer.py)
- [code/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/public_q24_native_transport_energy_final_workspace_v4/estimator.py)

## Evidence

- [optbeta_sweep_all_variants.log#L1](optbeta_sweep_all_variants.log#L1)
- [optbeta_panel_dep.log#L1](optbeta_panel_dep.log#L1)

<details>
<summary>All result files and records (2)</summary>

- [optbeta_sweep_all_variants.log](optbeta_sweep_all_variants.log) · log
- [optbeta_panel_dep.log](optbeta_panel_dep.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-02-variance-correction-scale-reduced-0-82-0-40-transferred) · [Copy provenance](../../../PROVENANCE.json)

