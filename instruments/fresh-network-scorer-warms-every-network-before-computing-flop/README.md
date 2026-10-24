# Fresh-network scorer warms every network before computing FLOP deltas

2026-10-20T14:46:13Z · tool · informational · measured

The first prediction binds network-specific frames and can add +0.2..+0.5% FLOPs. The scorer was corrected to warm each new network and repeat the base setting, while leaving raw accuracy comparisons unchanged.

Recorded measurement:

    - [fable-lead] 2026-10-20T14:46:13Z HARNESS LAW (eval_fresh_arm_v2 -> v3): the FIRST predict of an in-process evaluation pays the cold frame-binding bill (7100005: 2.299328e11 vs 2.289232e11 warm = +0.44%), so every v2 bill delta measured against a first-setting base was biased by -0.44% (JDIAG's true bill was ~+1.0%, not +0.57%); v3 runs a warm-up predict first. CMODE v1 was a NO-OP (JSON string keys vs the hook's int test; the identical U per setting proved it) -> candidate_cmode_sci_v2.py (29dad1a4) relaunched with v3 (run_cmode2_then_dmt_v1.sh, then the agent's DMT arm). HOMOGENEITY CHECK (homog_mode_v1.py): the exact common-mode covariance of a bias-free ReLU net (y = r f(u), Var(r) = 1/2048 -> Var(r) mu mu^T in Cpre) is 6-8% of mu^T Ct mu; the card misses only 1-5% of that term (c_obs = -0.01..-0.05 x Var(r)), i.e. the mu-direction deficit is a 0.3% relative error of the card's top covariance mode, consistent with dcov_structure_v3 (top-mode coefficient -0.0026..-0.0030 on net 35, -0.0006..-0.0020 on net 100). PROPAGATION (dcov_structure_v4.py): the top error eigenvector at layer l has |cos| 0.62-0.74 with the previous layer's top error mode pushed through W_l, and span(mu_l, pushed u0, u1) explains 10-19% of |dC_l|^2: the dominant late error is an accumulating mode born along mu at or before L8, drifting with depth; not a fixed card-known direction beyond its mu part.
    - [fable-lead] 2026-10-20T15:22:34Z HARNESS NOTE: in a multi-network in-process screen the FIRST predict of each NEW network binds that network's new frame shapes (bill +0.2..+0.5% on 7 of 10 store nets, identical for the +0.1 and -0.1 settings that follow), so per-setting bill deltas are artifacts unless the base setting is repeated after a warm-up on the same network; U deltas are unaffected (the debug and plain bases print identical U). The judge (one setting per process over 32 nets) is not affected: cmode_p10 on 8 judge ranks billed +0.052% vs the base on every rank (requests +314/net), raw +0.32% on 7 (lower 2/7).
    AUDIT: lead FD glob included crash logs (fixed to exact A/B/C + settings validation); step units differ (lead x must /0.2 before builder); old agent setting composer drops iso and attenuates previous dials (use clean helper/builder). Builder now uses cache-safe joint_sci_v2; release wrapper extraction fixed. Fresh harness v5 warmed only first net: repeated no-op settings expose -0.31..-0.37% spurious bill on second nets; v6 warms every net and prints 12-digit U. Provisional half-profile6: corrected adjusted +0.028% (3/6 lower), not a win. Existing compare32 reports percentage-point deltas against95c; new compare_relative_v1 gives true relative vs rel_v4a AND legacy. kc_late_m true adjusted -0.0843%, held25 -0.0962% (legacy -0.0830/-0.0937); below fire bar. Require both conventions <=-0.35 plus heldout negative.

## Programs

- [code/lead/runs/fable_oct19_kc_v1/eval_fresh_arm_v3.py](code/lead/runs/fable_oct19_kc_v1/eval_fresh_arm_v3.py)
- [code/lead/runs/fable_oct19_kc_v1/eval_fresh_arm_v6.py](code/lead/runs/fable_oct19_kc_v1/eval_fresh_arm_v6.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6187](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6187)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6192](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6192)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6218](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6218)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-10-20-fresh-network-scorer-warms-every-network-before-computing-flop) · [Copy provenance](../../PROVENANCE.json)

