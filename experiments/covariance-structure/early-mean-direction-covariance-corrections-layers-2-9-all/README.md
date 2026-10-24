# Early mean-direction covariance corrections at layers 2-9 all worsen MSE +0.4..+1.8%

2026-10-20T15:53:19Z · experiment · rejected · measured

Fixed positive and negative early covariance corrections failed on fresh networks 7100005 / 7100006. The tested ranges were layers 4-7, 4-9 and 2-7.

Recorded measurement:

    - [fable-lead] 2026-10-20T15:53:19Z EARLY-LAYER CMODE DEAD (fresh 7100005/7100006: c=+-0.05/0.1 at L4-7, L4-9, L2-7 all +0.4..+1.8%); eval stopped. BOX: swap 20.8 of 21.5 GB used (sticky), free 35-37% with one eval + one lane: the judge guard now kills a lane within seconds of any swap growth; from here the judge lanes get the box alone (no in-process evals beside them).
    - [fable-lead] 2026-10-20T15:52:35Z PER-NET SIGN PREDICTORS, second pass (26 store nets): the GNN correction's direction (cos with mu, with the post mean, its common-mode projection; delta_mu_sign_v1.py) |corr| <= 0.23 with the common-mode slope; its alignment with the card's own third-order and fourth-order MEAN terms (delta_k3_sign_v1.py) |corr| <= 0.22; every LOO sign rule >= +0.10%. The per-network factor is invisible to the GNN as well. EARLY-LAYER CMODE (c at L4-7 / L4-9 / L2-7, fresh 7100005): every setting hurts (+0.5..+1.8%, the negative sign too +0.93%): the early covariance along mu is at a sharp optimum. YNT 3.5 at L13/14 was already judged flat on 10-19 (TEAM_LOG 5951/5953): lane stopped. NEW LANES (never FD-tested natively; tails on the v4a science copy): kc_p/kc_m (_GN_KC x1.2 / x0.8: the kappa_C fraction of the GNN mean shift's rank-2 covariance update), rc_p/rc_m (_GN_RC -0.06 / -0.10), gsk_p/gsk_m (_GSK 1.1 / 0.9); queues v13a/v13b, 2 lanes once the early eval exits.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)
- [code/lead/runs/fable_oct19_kc_v1/candidate_cmode_sci_v3.py](code/lead/runs/fable_oct19_kc_v1/candidate_cmode_sci_v3.py)
- [code/lead/runs/fable_oct19_kc_v1/eval_fresh_arm_v5.py](code/lead/runs/fable_oct19_kc_v1/eval_fresh_arm_v5.py)
- [code/lead/runs/fable_oct19_kc_v1/make_cmode_v3.py](code/lead/runs/fable_oct19_kc_v1/make_cmode_v3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6197](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6197)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6196](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6196)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-early-mean-direction-covariance-corrections-layers-2-9-all) · [Copy provenance](../../../PROVENANCE.json)

