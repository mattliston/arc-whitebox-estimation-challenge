# Third-order plane target made consistent with learned mean shifts: +34..+74% MSE at full strength

2026-10-20T15:02:04Z · experiment · rejected · measured

Correcting the third-order pair-plane target for the learned mean shift broke the co-adapted fit. Negative-strength tests also split by network, and updating births alone worsened mean MSE.

Recorded measurement:

    - [fable-lead] 2026-10-20T15:02:04Z AGENT DMT ARM DEAD (agent_ideas_v1/candidate_dmt_sci_v1.py: the (2,1) post-plane target made consistent with the GNN mean shift dm, strength S; HOST = births see the corrected mean; eval_fresh_arm_v3, 4 fresh nets): S=1.0 U +34..+74%, S=0.5 +4.6..+22.8%, HOST alone -0.04/+0.83/+0.57/+0.51% (mean +0.46%), S=1+HOST +34..+73%. The target's pre-dm mean is part of the co-adapted fit; the consistent term explodes (same law as the truth-fitted cumulant corrections).
    - [fable-lead] 2026-10-20T15:44:12Z DMT NEGATIVE SIDE (agent's recipe; fresh 7100005/7100006): S=-0.15 -> U -1.66% / +0.76%, S=-0.3 -> -1.42% / +3.18% (mean -0.46% / +0.87%): the same per-network split as CMODE (7100005 wants every rank-one correction one way, 7100006 the other); dead as a fixed arm. Board 2026-10-20T15:44:12Z: see astra_board dir above.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/agent_ideas_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_dmt_sci_v1.py](code/candidate_dmt_sci_v1.py)
- [code/mean_origin_v1.py](code/mean_origin_v1.py)
- [code/run_j5_v1.sh](code/run_j5_v1.sh)
- [code/candidate_j5join_sci_v1.py](code/candidate_j5join_sci_v1.py)
- [code/run_evals_v1.sh](code/run_evals_v1.sh)
- [code/cmode_magnitude_v1.py](code/cmode_magnitude_v1.py)
- [code/cpre_birth_v1.py](code/cpre_birth_v1.py)
- [code/joint_fd_analysis_v1.py](code/joint_fd_analysis_v1.py)
- [code/cmode_rules_v2.py](code/cmode_rules_v2.py)
- [code/cmode_ng_split_v1.py](code/cmode_ng_split_v1.py)
- [code/cmode_injection_v1.py](code/cmode_injection_v1.py)
- [code/census_reweight_v1.py](code/census_reweight_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6189](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6189)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6195](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6195)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [IDEAS_v1.md](../../join-sketch/per-row-join-frame-energy-repair-increases-mse-0/IDEAS_v1.md) · record
- [IDEAS_v2.md](../../join-sketch/per-row-join-frame-energy-repair-increases-mse-0/IDEAS_v2.md) · record
- [joint_step_settings_v1.json](../../join-sketch/per-row-join-frame-energy-repair-increases-mse-0/joint_step_settings_v1.json) · result
- [IDEAS_v3.md](../../join-sketch/per-row-join-frame-energy-repair-increases-mse-0/IDEAS_v3.md) · record
- [census_a7ffd713_weighted_v1.json](../../join-sketch/per-row-join-frame-energy-repair-increases-mse-0/census_a7ffd713_weighted_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-third-order-plane-target-made-consistent-learned-mean-shifts) · [Copy provenance](../../../PROVENANCE.json)

