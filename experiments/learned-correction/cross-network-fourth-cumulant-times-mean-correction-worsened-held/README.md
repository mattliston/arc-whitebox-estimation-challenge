# Cross-network fourth-cumulant times mean correction worsened held-out error capture

2026-08-30 · experiment · informational · measured

k4xm profile applied to the HONEST planes (each net corrected by the other's frozen profile, betas refit held-out): 12.4%/8.9% - WORSE than plain honest (13.6%/10.7%). The plane-metric gain does not convert through the chain even with refit weights (echoes carrier-plane-rms-is-not-the-objective). NIGHT ENDSTATE: deployable-class honest = 13.6/10.7% held-out cross-net = 2.374e-8/2.390e-8 (validation frame; deploy-frame honest rerun pending).

Recorded measurement:

    k4xm profile applied to the HONEST planes (each net corrected by the other's frozen profile, betas refit held-out): 12.4%/8.9% - WORSE than plain honest (13.6%/10.7%). The plane-metric gain does not convert through the chain even with refit weights (echoes carrier-plane-rms-is-not-the-objective). NIGHT ENDSTATE: deployable-class honest = 13.6/10.7% held-out cross-net = 2.374e-8/2.390e-8 (validation frame; deploy-frame honest rerun pending). The 1e-8 path: (a) derive the k4xm term's true injection, (b) 10-net LOO beta infra, (c) deploy-frame honest gate, (d) hub magnitude+CP compression for submission-time supply.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g19_attribution_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_beta_transfer.py](code/run_beta_transfer.py)
- [code/run_k4m_correction.py](code/run_k4m_correction.py)

## Evidence

- [beta_transfer_k4mhon.log#L1](beta_transfer_k4mhon.log#L1)

<details>
<summary>All result files and records (1)</summary>

- [beta_transfer_k4mhon.log](beta_transfer_k4mhon.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-30-cross-network-fourth-cumulant-times-mean-correction-worsened-held) · [Copy provenance](../../../PROVENANCE.json)

