# Eight-network correction-chain validation: 18.7% mean held-out capture, all folds positive

2026-08-31 · experiment · informational · measured

The full-tensor correction chain was evaluated by leave-one-network-out validation on eight networks. Every fold showed positive capture, but this historical truth-fed replay was a capacity measurement rather than a deployable source.

Recorded measurement:

    {
     "35": {
      "betas": [
       0.028,
       0.218,
       0.563
      ],
      "capture_pct": 12.9,
      "mse_nc": 2.3297878569748545e-08
     },
     "100": {
      "betas": [
       0.058,
       0.23,
       0.57
      ],
      "capture_pct": 7.5,
      "mse_nc": 2.541657931553502e-08
     },
     "150": {
      "betas": [
       0.004,
       0.195,
       0.563
      ],
      "capture_pct": 18.4,
      "mse_nc": 2.5353801677106782e-08
     },
     "200": {
      "betas": [
       0.032,
       0.226,
       0.539
      ],
      "capture_pct": 19.0,
      "mse_nc": 3.345201479568237e-08
     },
     "250": {
      "betas": [
       0.039,
       0.213,
       0.546
      ],
      "capture_pct": 20.5,
      "mse_nc": 2.238461671282292e-08
     },
     "300": {
      "betas": [
       0.044,
       0.208,
       0.526
      ],
      "capture_pct": 27.2,
      "mse_nc": 2.4606793783877718e-08
     },
     "400": {
      "betas": [
       0.022,
       0.204,
       0.542
      ],
      "capture_pct": 22.6,
      "mse_nc": 2.6327855740074894e-08
     },
     "500": {
      "betas": [
       0.016,
       0.217,
       0.541
      ],
      "capture_pct": 21.5,
      "mse_nc": 2.958321806570241e-08
     },
     "mean_capture": 18.7,
     "sd_capture": 5.7
    }
    

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g19_attribution_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_loo_grade.py](code/run_loo_grade.py)

## Evidence

- [LOO_GRADE_tensor.json#L1](LOO_GRADE_tensor.json#L1)

<details>
<summary>All result files and records (1)</summary>

- [LOO_GRADE_tensor.json](LOO_GRADE_tensor.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-31-eight-network-correction-chain-validation-18-7-mean-held) · [Copy provenance](../../../PROVENANCE.json)

