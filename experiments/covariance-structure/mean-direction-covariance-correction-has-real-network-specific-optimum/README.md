# Mean-direction covariance correction has a real network-specific optimum but no usable sign predictor

2026-10-20T14:55:25Z · experiment · rejected · measured

A rank-one covariance correction along the pre-activation mean changed MSE by network-specific signs. On 24 complete plus one partial store network, predeclared rules failed leave-one-network-out evaluation and an ideal sign oracle was worth only -0.53% pooled.

Recorded measurement:

    - [fable-lead] 2026-10-20T14:55:25Z CMODE v2 (candidate_cmode_sci_v2.py 29dad1a4: Cpre += c mean(v)/|mu|^2 mu mu^T at L8-14; eval_fresh_arm_v3, 4 fresh nets, U = final cross-half MSE): per net c=0.05/0.1/0.2/-0.1 -> 7100005 +0.30/+0.96/+1.66/-0.22%, 7100006 +0.10/+0.11/+0.71/+0.51%, 7100007 -0.32/-0.24/+0.42/+1.06%, 7100008 -0.26/-0.52/-1.48/+0.85%; L10-14 only c=0.1: +0.41/-0.03/+0.05/-0.34%. Mean of the fixed coefficient ~0 (c=0.05: -0.04%); the per-net MSE-optimal c spans -0.1..>+0.2 with gains of 0.2-1.5% each = a per-network lever IF the sign is predictable from an observable (next: a store-net screen with the stateprobe features). Bill moves +-0.3% per net via roster responses.
    - [fable-lead] 2026-10-20T15:26:36Z CMODE SIGN STUDY, 12 of 24 store nets (cmode_sign_fit_v3.py; target = slope of U(c) at 0, pre-declared features Q = mumode/radial and the late kink fraction, exploratory = the stateprobe summaries): per-net slopes span -10..+8 (%/unit c); NO rule predicts the sign LOO (Q +0.27% mean, kink +0.18%, all features +0.28%; permutation chance 0.6-0.95); oracle (best of +-0.1 per net) -0.38% mean / -0.37% pooled = the line's ceiling even with a perfect rule; always +0.1 = +0.24%, always -0.1 = +0.06%; corr(base U, slope) +0.53 on store nets (hard store nets want c < 0, the OPPOSITE of the fresh-net pattern): the per-net factor is not an observable. PUSH-FACTORING dead end (code reading): each born pair has its own base column and its own hub unit (newborn = (q, selected, base(n x births), t, s); legs = base + t e_hub, base + s e_hub), so at ages >= 2 the pair's two pushed legs span exactly two independent columns (base' and the pushed hub direction (W_{b+1}..W_l)[hub]): a factored push saves nothing (the age-1 newborn push already uses the base + W-row form).
    - [fable-lead] 2026-10-20T15:38:57Z CMODE SIGN STUDY FINAL (24 store nets + 1 partial; cmode_sign_fit_v3.py): per-net slopes of U(c) at c=0 span -19..+18 %/unit; pre-declared rules fail LOO (Q +0.20%, kink fraction +0.20%, both +0.46%; permutation chance 0.9-1.0); the all-feature LOO rule reaches only -0.10% (lower 16/25, chance 0.06); oracle (best of +-0.1 per net) -0.54% mean / -0.53% pooled; always +0.1 = +0.31%, always -0.1 = +0.06%; corr(base U, slope) +0.24. VERDICT: the common-mode correction is a real per-network effect with no observable predictor; as a fixed or rule-driven arm it is DEAD (ceiling -0.5% even with a perfect oracle). Remaining cheap checks queued for the record only: cmode +-0.1 on the 32 judge ranks (relaunch), DMT negative side, cmode at the early layers 4-9.
    - [fable-lead] 2026-10-20T15:52:35Z PER-NET SIGN PREDICTORS, second pass (26 store nets): the GNN correction's direction (cos with mu, with the post mean, its common-mode projection; delta_mu_sign_v1.py) |corr| <= 0.23 with the common-mode slope; its alignment with the card's own third-order and fourth-order MEAN terms (delta_k3_sign_v1.py) |corr| <= 0.22; every LOO sign rule >= +0.10%. The per-network factor is invisible to the GNN as well. EARLY-LAYER CMODE (c at L4-7 / L4-9 / L2-7, fresh 7100005): every setting hurts (+0.5..+1.8%, the negative sign too +0.93%): the early covariance along mu is at a sharp optimum. YNT 3.5 at L13/14 was already judged flat on 10-19 (TEAM_LOG 5951/5953): lane stopped. NEW LANES (never FD-tested natively; tails on the v4a science copy): kc_p/kc_m (_GN_KC x1.2 / x0.8: the kappa_C fraction of the GNN mean shift's rank-2 covariance update), rc_p/rc_m (_GN_RC -0.06 / -0.10), gsk_p/gsk_m (_GSK 1.1 / 0.9); queues v13a/v13b, 2 lanes once the early eval exits.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_cmode_sci_v2.py](code/candidate_cmode_sci_v2.py)
- [code/cmode_sign_fit_v3.py](code/cmode_sign_fit_v3.py)
- [code/delta_mu_sign_v1.py](code/delta_mu_sign_v1.py)
- [code/delta_k3_sign_v1.py](code/delta_k3_sign_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6188](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6188)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6193](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6193)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6194](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6194)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6196](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6196)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-mean-direction-covariance-correction-has-real-network-specific-optimum) · [Copy provenance](../../../PROVENANCE.json)

