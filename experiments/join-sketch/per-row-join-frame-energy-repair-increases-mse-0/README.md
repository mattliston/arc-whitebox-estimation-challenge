# Per-row join-frame energy repair increases MSE +0.19% / +0.04% and adds about 1.0% FLOPs

2026-10-20T14:17:29Z · experiment · rejected · measured

The rank-384 join frame already captured 99.995% of joined carrier energy. Restoring row energies did not restore the missing subspace orientation, and warm-up correction revised the apparent FLOP cost upward.

Recorded measurement:

    - [fable-lead] 2026-10-20T14:17:29Z RESUMED after a 3 h idle gap (Matt: 'why did you stop'; the idle was my error: no waiting for his word, keep generating arms). Board 14:11Z: marius 1.30137e-9, oliver 1.35922e-9, J2W 1.41601e-9, us #337259 1.42847e-9 (4th). CLOSED today: (1) Monte Carlo of any kind as the leaders' mechanism: final-layer post variances average 0.085 (mc_post_var, fresh 7100000), so 6,372 samples (our whole bill) give MSE 1.3e-5 = 1,000x the card; (2) the K22 plane scale arms (candidate_k22s_*: built 10-19, in-process x1.5 at L12-14 +979%, x0.5 +283%: a co-adapted recursion); (3) JDIAG (candidate_jdiag_sci_v1.py 37a375ef, make_jdiag_v1.py: per-row energy repair of the join frame Qn -> diag(G) exact): the rank-384 frame already captures 99.995% of the joined carrier energy (L10 width 5440 x 832 rows, trace full 2090.7 vs projected 2090.6; per-row ratio 0.96..1.003), so the repair moves nothing: fresh 7100005/7100007 U +0.19%/+0.04% for bill +0.57% (M = Fj w Fj^T over 5,440 columns) = DEAD; the join's MSE cost (no-join ceiling -1.55%) lives in the 0.005% residual's orientation (the P G (I-P) cross term the eigen frame kills), i.e. a subspace problem the record priced at 1:1 (NT). (4) Lanes: r7_320 / r7_352 (layer-7 join rank 320/352 tails on the v4a science copy; the 10-18 one-net read at 320 was bill -0.84% raw +0.61% adjusted -0.23%, never banked for residual-margin reasons that Matt now says are solved) launched 14:13Z on 2 lanes vs science32_rel_v4a_m15. Idea agent dissect337259 spawned 14:16Z (Matt's ask) to take #337259 apart for MSE/bill levers; report -> lead/runs/fable_oct19_kc_v1/agent_ideas_v1/IDEAS_v1.md.
    - [fable-lead] 2026-10-20T14:46:13Z HARNESS LAW (eval_fresh_arm_v2 -> v3): the FIRST predict of an in-process evaluation pays the cold frame-binding bill (7100005: 2.299328e11 vs 2.289232e11 warm = +0.44%), so every v2 bill delta measured against a first-setting base was biased by -0.44% (JDIAG's true bill was ~+1.0%, not +0.57%); v3 runs a warm-up predict first. CMODE v1 was a NO-OP (JSON string keys vs the hook's int test; the identical U per setting proved it) -> candidate_cmode_sci_v2.py (29dad1a4) relaunched with v3 (run_cmode2_then_dmt_v1.sh, then the agent's DMT arm). HOMOGENEITY CHECK (homog_mode_v1.py): the exact common-mode covariance of a bias-free ReLU net (y = r f(u), Var(r) = 1/2048 -> Var(r) mu mu^T in Cpre) is 6-8% of mu^T Ct mu; the card misses only 1-5% of that term (c_obs = -0.01..-0.05 x Var(r)), i.e. the mu-direction deficit is a 0.3% relative error of the card's top covariance mode, consistent with dcov_structure_v3 (top-mode coefficient -0.0026..-0.0030 on net 35, -0.0006..-0.0020 on net 100). PROPAGATION (dcov_structure_v4.py): the top error eigenvector at layer l has |cos| 0.62-0.74 with the previous layer's top error mode pushed through W_l, and span(mu_l, pushed u0, u1) explains 10-19% of |dC_l|^2: the dominant late error is an accumulating mode born along mu at or before L8, drifting with depth; not a fixed card-known direction beyond its mu part.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_jdiag_sci_v1.py](code/candidate_jdiag_sci_v1.py)
- [code/make_jdiag_v1.py](code/make_jdiag_v1.py)
- [code/agent_ideas_v1/mean_origin_v1.py](code/agent_ideas_v1/mean_origin_v1.py)
- [code/agent_ideas_v1/run_j5_v1.sh](code/agent_ideas_v1/run_j5_v1.sh)
- [code/agent_ideas_v1/candidate_dmt_sci_v1.py](code/agent_ideas_v1/candidate_dmt_sci_v1.py)
- [code/agent_ideas_v1/candidate_j5join_sci_v1.py](code/agent_ideas_v1/candidate_j5join_sci_v1.py)
- [code/agent_ideas_v1/run_evals_v1.sh](code/agent_ideas_v1/run_evals_v1.sh)
- [code/agent_ideas_v1/cmode_magnitude_v1.py](code/agent_ideas_v1/cmode_magnitude_v1.py)
- [code/agent_ideas_v1/cpre_birth_v1.py](code/agent_ideas_v1/cpre_birth_v1.py)
- [code/agent_ideas_v1/joint_fd_analysis_v1.py](code/agent_ideas_v1/joint_fd_analysis_v1.py)
- [code/agent_ideas_v1/cmode_rules_v2.py](code/agent_ideas_v1/cmode_rules_v2.py)
- [code/agent_ideas_v1/cmode_ng_split_v1.py](code/agent_ideas_v1/cmode_ng_split_v1.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/agent_ideas_v1/cmode_injection_v1.py](code/agent_ideas_v1/cmode_injection_v1.py)
- [code/agent_ideas_v1/census_reweight_v1.py](code/agent_ideas_v1/census_reweight_v1.py)

</details>

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6184](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6184)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6187](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6187)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [IDEAS_v1.md](IDEAS_v1.md) · record
- [IDEAS_v2.md](IDEAS_v2.md) · record
- [joint_step_settings_v1.json](joint_step_settings_v1.json) · result
- [IDEAS_v3.md](IDEAS_v3.md) · record
- [census_a7ffd713_weighted_v1.json](census_a7ffd713_weighted_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-per-row-join-frame-energy-repair-increases-mse-0) · [Copy provenance](../../../PROVENANCE.json)

