# Pair-series terms 25–30: legal cost but final mean change at most 1.2566e-8 RMS

2026-08-25 · experiment · rejected · measured

Adjudicated at design stage by exact target-free measurement, with no data-bearing launch. The additive t=25..30 pair-series tail is cost-admissible (live added FLOPs 36,370,219,020 against the 67,331,117,831 headroom; synthetic billing 35,948,494,848 within 1.2%) and numerically stable on all eight probe rows including hard rows 201 and 643 and repaired tail 55 - unlike the closed full-Q32 rebuild - but it is effect-capped: the tail changes the reported means by a median 1.6e-5 of the final error scale (max final-layer delta RMS 1.2566e-8 versus population error RMS 3.0193e-4), bounding any aligned mean squared error (MSE) effect near 8e-5 of MSE. The truncated Mehler tail is not where the host error lives; the S receipt's t=24 born-error correlation was co-location through bond strength, not causation.

Recorded measurement:

    Adjudicated at design stage by exact target-free measurement, with no data-bearing launch. The additive t=25..30 pair-series tail is cost-admissible (live added FLOPs 36,370,219,020 against the 67,331,117,831 headroom; synthetic billing 35,948,494,848 within 1.2%) and numerically stable on all eight probe rows including hard rows 201 and 643 and repaired tail 55 - unlike the closed full-Q32 rebuild - but it is effect-capped: the tail changes the reported means by a median 1.6e-5 of the final error scale (max final-layer delta RMS 1.2566e-8 versus population error RMS 3.0193e-4), bounding any aligned MSE effect near 8e-5 of MSE. The truncated Mehler tail is not where the host error lives; the S receipt's t=24 born-error correlation was co-location through bond strength, not causation. The quartic-star birth marker (+0.335) survives and points at the Edgeworth weight expansion order, whose pair weights stop at 1 + L3 + L4 + L3^2/2.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_host_tail_extension_probe_v1/estimator_t30.py](code/public_host_tail_extension_probe_v1/estimator_t30.py)
- [code/public_host_tail_extension_probe_v1/run_ab_probe.py](code/public_host_tail_extension_probe_v1/run_ab_probe.py)
- [code/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/estimator.py](code/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/estimator.py)

## Evidence

- [AB_EFFECT_PROBE.json#L1](AB_EFFECT_PROBE.json#L1)
- [SYNTHETIC_EINSUM_BILLING.json#L1](SYNTHETIC_EINSUM_BILLING.json#L1)
- [T_DESIGN_CLOSURE.json#L48](T_DESIGN_CLOSURE.json#L48)

<details>
<summary>All result files and records (3)</summary>

- [AB_EFFECT_PROBE.json](AB_EFFECT_PROBE.json) · result
- [SYNTHETIC_EINSUM_BILLING.json](SYNTHETIC_EINSUM_BILLING.json) · result
- [T_DESIGN_CLOSURE.json](T_DESIGN_CLOSURE.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-25-pair-series-terms-2530-legal-cost-but-final-mean) · [Copy provenance](../../../PROVENANCE.json)

