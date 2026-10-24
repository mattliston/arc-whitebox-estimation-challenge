# Native covariance oracles reduce late-layer error less than the stale research estimator predicted

2026-10-20T03:40:21Z · finding · informational · measured

Native substitution of exact pre-covariance at layers 10-14 reduced final MSE by -26.2 / -33.9% on networks 35 / 100; exact per-unit pre-variance gave -17.7 / -25.6%. The older research estimator overstated truth-plane headroom by 4-6x, so subsequent decisions used native oracles.

Recorded measurement:

    - [fable-lead] 2026-10-20T03:40:21Z NATIVE ORACLE ANATOMY (shipped physics candidate_rel_nosync_sci_v1 = #337212/#337247, in-process on store nets 35 / 100 vs truth688, FINAL-layer MSE; runners lead/runs/fable_oct19_kc_v1/native_cov_oracle_v1.py (pair-map inputs/outputs replaced by the 32M mcall truths) and native_k31_oracle_v1.py (truth planes through the GN covariance kernel)). Base 10.969e-9 / 12.731e-9. Exact pre-mean at L14: -41% / -39%; exact pre-cov at L14: -8.7 / -15.2; L13: -9.0 / -14.7; L12: -7.1 / -18.2; L10-14: -26.2 / -33.9; exact per-unit pre-VARIANCE only at L10-14: -17.7 / -25.6 (L14 alone +0.1 / -7.0); exact mean+cov at L14: -54 / -54. Planes in the covariance output: truth K21 at L10-14: -5.6 / -8.4 (L8-14 -0.4 / -8.4; L2-14 +9.1 / +4.0 = early truth HURTS, co-adapted); truth K31 remainder L12-14: -2.0 / -2.6 (half strength -3.0 / -2.9), L10-14 -2.8 / +3.2; K21+K31 L12-14: -5.5 / -10.1. Frobenius splits of the K21 deficit (scale-mode column q_i mu_j, rest, global scale x1.34, row scale) each give +90..+440% alone: the -5.6% is a small residual of large cancelling covariance effects (effect metric, not Frobenius). LAW: the torch research card (physicist/e16b_f4b.py base c=0.75,age=2,f4=0.6) is the 10-02 S38 physics, 16-19% worse than the shipped card on the same nets; its truth-plane oracles (-18% at L10-14) overstate native headroom 4-6x. Decide on native oracles only. Per-unit cumulants natively at L12: card k3 = 0.84-0.86 x truth (corr .96-.97), card k4 = 0.91 x truth (corr .80-.85).

## Programs

- [native_cov_oracle_v1.py](native_cov_oracle_v1.py)
- [native_k31_oracle_v1.py](native_k31_oracle_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6126](../../TEAM_LOG_2026-10-01_to_10-22.md#L6126)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-native-covariance-oracles-reduce-late-layer-error-less-than) · [Copy provenance](../../../PROVENANCE.json)

