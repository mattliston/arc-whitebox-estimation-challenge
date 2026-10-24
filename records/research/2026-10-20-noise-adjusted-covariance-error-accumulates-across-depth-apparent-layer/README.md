# Noise-adjusted covariance error accumulates across depth; the apparent layer-four onset is not established

2026-10-20T16:16:36Z · finding · informational · measured

Early probes initially attributed the covariance deficit to layer four. Subsequent noise-adjusted attribution corrected this to roughly uniform accumulation across depth, with a growing mean-aligned component and network-specific compensation.

Recorded measurement:

    - [fable-lead] 2026-10-20T16:16:36Z COVARIANCE ERROR BIRTH LAYER (probe_cpre_dump_v1.py, nets 35/100, card Cpre vs 32M-sample truth, all layers): |dC|/|Ct| = 0.36/0.32/0.30% at L1-3 = the truth's own MC noise floor (a 32M-sample covariance has ~0.4% relative Frobenius noise; the card's L1 is exact), then 0.31/0.33/0.36/0.39/0.43% at L4-8 and 0.63% at L14: the card's real error is ~0 through L3, ~0.1% at L4-5, 0.24% at L8, 0.52% at L14 (noise removed in quadrature). The top error mode appears at L4 on BOTH nets (share 3.5% / 3.1%, sign turns negative, |cos(u0,mu)| jumps 0.21 -> 0.73) and grows every layer (7% L5, 12% L6, 16% L7, 18-27% late): the common-mode deficit is BORN AT LAYER 4 and accumulates; it is not a late-layer phenomenon and not an early (L1-3) one. L4 is where the first cohorts pass the age-2 window of the local fourth-order terms (K4L_AGE/F4C_AGE/NP_K3L_AGE = 2; the ages 3-4 family K4L34 exists only at the inner layers): hypothesis = the deficit is the expired fourth-order content of cohorts aged 3+ at L4-7.
    - [fable-lead] 2026-10-20T16:20:27Z COVARIANCE INJECTION (cov_injection_pre_v1.py, pre dumps + first-order gate transmission + exact push, nets 35/100): the per-layer decomposition is dominated by the truth's MC noise (|d| 0.14-0.19 Frobenius at every layer incl. the exact L1-2), so sizes are not readable; the mu-mode coefficients are: injected along mu per layer -0.005..-0.024 (dimensionless) at L2-8, ~0 after L8; the pushed (propagated) mu-mode coefficient grows -0.001 (L2) -> -0.02 (L5) -> -0.10 (L8) -> -0.24 (L14, net 35) while the common mode itself grows (mu^T C mu/(|mu|^2 mean v) 15 -> 42 -> 89): the RELATIVE deficit of the common-mode variance is ~0 at L3, -0.15% at L5 and a steady -0.25..-0.3% from L8 on. A gentle early correction (parabola through c = +-0.05 at L4-7 on 7100005: c* = +0.007, gain -0.015%) is worthless. Common-mode story closed: a ~0.3% relative under-estimate of the mean-direction covariance established by L8, whose MSE effect is +-1% per net with a net-specific sign.
    - [fable-lead] 2026-10-20T16:26:26Z AGENT B7 (IDEAS_v2.md; cmode_injection_v1.log): the common-mode covariance deficit is INJECTED at L2-7 at a flat ~-0.1% of the mode per layer (already at L2-3, before any cohort reaches age 3; independent of the expired-column count), inherited only from L8 on; 20-40% of the injection is the GNN's rank-2 covariance update C -= (0.92 - KC)(m dm^T + dm m^T + dm dm^T) (the GNN raises means early, lowers them late, which is why the deficit stops growing late). Mechanism-based arms: (1) kc_early_p (KC x1.3 at L1-7 = less subtraction) moved up the queue; (2) EARLY CMODE PROFILE at the truth-implied magnitude c = {2: .005, 3: .008, 4: .015, 5: .03, 6: .05, 7: .075} and its half, added to the fresh-net gap check (my earlier +-0.05/0.1 at L4-7 was a 2-5x overshoot).
    - [fable-lead] 2026-10-20T16:38:00Z kc_m (kappa_C x0.8) FLAT on 32 ranks: +0.010% (se 0.082, lower 13/32); the global KC knob is at its optimum (x1.2 +0.08%). POST-COVARIANCE PROBES (probe_cpost_dump_v1.py, nets 35/100): the common-mode deficit m^T dC m / m^T Ct m is -0.13% already at L1 (post of the exact Gaussian layer: the pair map's own output), -0.16% L2-3, -0.23..-0.35% L4-7 on net 35 then -0.25..-0.34% through L13; on net 100 it peaks -0.22% at L6 and shrinks to -0.06% by L13 (the late corrections compensate on 100, not on 35: net-specific). Agent B8 (IDEAS_v2): along the common mode the Gaussian pair series is exact; the NON-GAUSSIAN transfer (cumulant-carried: b-carrier, roster K21 cross terms, k22) runs 0.5-1.4% short at L1-8 and matches truth from L9 (where the edge/GN knobs were fitted); budget of the early injection: cumulant transfer 40-60%, GNN rank-2 (KC) 20-40%, GN dcov 15-20%. Prediction: a per-layer GSK split reads gsk_early_p NEGATIVE, gsk_late_p ~0. BUILT: _GSK per layer via a _gq wrapper keyed on _PHYS_LAYER (tails gsk_early_p 0.99 / gsk_early_m 0.81 at L1-7, gsk_late_p 0.99 at L8-14; base _GSK = 0.9, so the earlier global gsk arms were rebuilt: p 1.0 / m 0.8). Queue now: kc_late_m (running), kc_late_p, kc_early_p, gsk_early_p, iso_early_m, kc_early_m, gsk_early_m, gsk_late_p, gsk_p, gsk_m, rc_m, rc_p, then the remaining early arms; the early CMODE profiles (full/half) run in the gap after kc_late_m.
    Later correction: CORRECTION 2026-10-20 16:2xZ (agent dissect337259, IDEAS_v2.md): the covariance error is NOT born at L4: noise-subtracted (32M-sample truth Frobenius noise 0.35% at L1 falling to 0.13% at L14), the real card error is 0.02% at L1 rising by a straight +0.045% per layer to 0.6% at L14 on both nets (uniform accrual, no jump at joins/edges/L4); the top mode's growing share is accumulation along mu. The common mode along mu is 30-45% of the REAL late error; truth-implied CMODE coefficients differ 2-5x between nets 35 and 100 (same sign) and the per-net magnitude is unpredictable (sign predictable at 2-5% chance via the depth trend of the GNN correction magnitude, corr -0.52, but the best realizable rule is -0.09%; perfect sign at |c| = 0.2 = -0.67%; quadratic oracle -0.93%). Final-mean error origin on 40 store nets: L0-7 8.4%, L8-12 39.6%, L13-15 51.9%; propagated mean errors decay ~16%/layer (early levers discounted ~7x). Early-layer dial arms are therefore low value; the early (2,1) reads are essential (NOREAD x11-12) and not subsamplable.

## Programs

- [probe_cpre_dump_v1.py](probe_cpre_dump_v1.py)
- [cov_injection_pre_v1.py](cov_injection_pre_v1.py)
- [probe_cpost_dump_v1.py](probe_cpost_dump_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6201](../../TEAM_LOG_2026-10-01_to_10-22.md#L6201)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6202](../../TEAM_LOG_2026-10-01_to_10-22.md#L6202)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6203](../../TEAM_LOG_2026-10-01_to_10-22.md#L6203)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6204](../../TEAM_LOG_2026-10-01_to_10-22.md#L6204)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-noise-adjusted-covariance-error-accumulates-across-depth-apparent-layer) · [Copy provenance](../../../PROVENANCE.json)

