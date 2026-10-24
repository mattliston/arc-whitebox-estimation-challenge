# Additional first join at layer 5: -0.95% adjusted on one fresh network and +2.7% on another

2026-10-20T15:05:06Z · experiment · rejected · measured

Absorbing the oldest cohort at layer 5 cost more sketch, factorization and frame-reconstruction work than it saved in young transport. Accuracy changed by opposite signs on the two fresh networks.

Recorded measurement:

    - [fable-lead] 2026-10-20T15:05:06Z AGENT J5 (extra first join at layer 5 absorbing cohort 0 at age 5; agent_ideas_v1/candidate_j5join_sci_v1.py; v3 harness, warm): 7100005 U 1.3042e-8 vs 1.3242e-8 = -1.51% MSE, bill +0.57% (adjusted -0.95%); 7100006 U 1.3367e-8 vs 1.3104e-8 = +2.01% MSE, bill +0.70% (adjusted +2.7%). NOT a bill cut (the L5 sketch/QR/core and the uncached L5-6 frame reconstruct cost more than cohort 0's young pushes save) and a +-2% per-net MSE swing: dead as built. PATTERN (agent's reading): the per-net sign of three different rank-one changes (CMODE c*, DMT S*, J5) orders the same way (7100005 'over-covarianced' wants c<0; 7100008, the hardest net, wants c>0): a common per-net factor; the store screen (24 nets, running) decides whether an observable predicts it.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/agent_ideas_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_j5join_sci_v1.py](code/candidate_j5join_sci_v1.py)
- [code/mean_origin_v1.py](code/mean_origin_v1.py)
- [code/run_j5_v1.sh](code/run_j5_v1.sh)
- [code/candidate_dmt_sci_v1.py](code/candidate_dmt_sci_v1.py)
- [code/run_evals_v1.sh](code/run_evals_v1.sh)
- [code/cmode_magnitude_v1.py](code/cmode_magnitude_v1.py)
- [code/cpre_birth_v1.py](code/cpre_birth_v1.py)
- [code/joint_fd_analysis_v1.py](code/joint_fd_analysis_v1.py)
- [code/cmode_rules_v2.py](code/cmode_rules_v2.py)
- [code/cmode_ng_split_v1.py](code/cmode_ng_split_v1.py)
- [code/cmode_injection_v1.py](code/cmode_injection_v1.py)
- [code/census_reweight_v1.py](code/census_reweight_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6191](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6191)

<details>
<summary>All result files and records (6)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [IDEAS_v1.md](../per-row-join-frame-energy-repair-increases-mse-0/IDEAS_v1.md) · record
- [IDEAS_v2.md](../per-row-join-frame-energy-repair-increases-mse-0/IDEAS_v2.md) · record
- [joint_step_settings_v1.json](../per-row-join-frame-energy-repair-increases-mse-0/joint_step_settings_v1.json) · result
- [IDEAS_v3.md](../per-row-join-frame-energy-repair-increases-mse-0/IDEAS_v3.md) · record
- [census_a7ffd713_weighted_v1.json](../per-row-join-frame-energy-repair-increases-mse-0/census_a7ffd713_weighted_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-additional-first-join-layer-5-0-95-adjusted-one) · [Copy provenance](../../../PROVENANCE.json)

