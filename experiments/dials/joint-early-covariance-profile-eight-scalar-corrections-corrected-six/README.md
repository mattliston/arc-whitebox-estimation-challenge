# Joint early covariance profile and eight scalar corrections: corrected six-network baseline +0.02763%

2026-10-20T18:16:42Z · experiment · rejected · measured

Joint finite differences around the early covariance profile found no large interaction gain. Corrected paired results were +0.02763% for the half profile and +0.11878% for the full profile; independent-dial extrapolations were not treated as validated candidates.

Recorded measurement:

    - [fable-lead] 2026-10-20T18:16:42Z JOINT FD RESULT (6 fresh nets 7100005-10; joint_fd_analysis_v1.py + agent_ideas_v1/joint_fd_analysis_v1.py; logs eval_joint_fd_{A,B,C}.log): half early CMODE profile (B1) +0.004% mean (lower 3/6; per net +0.38/-0.60/-0.18/+0.08/+0.55/-0.20), full (B2) +0.095%: regime C (co-adapted/undecided), pooled parabola c* 0.23 profile units. Dial slopes per +-20% step at B0: iso_early -0.25% (4/6), kc_early -0.14% (4/6), edge_late -0.09% (5/6), k21n_early -0.09%, k31 ~0, iso_late +0.03%; at B1 the early slopes relax (substitutes of the profile) and k31 becomes -0.15% (4/6); gsk_early/gk4_early were no-ops in this run (cache; fixed afterwards). Independent-dial Newton sums: micro-stack at B0 -0.13%, profile + re-adaptation at B1 -0.30% (se of each dial ~ its effect); the agent's damped block-scaled joint step is EMPTY (no dial clears its threshold). Pair cross terms: kc_early+ & edge_late- -0.08 (se 0.22), edge_late- & k31- -0.08 (se 0.16), others 0. VERDICT: moving the co-adapted early/late dials together around the profile yields at most a ~0.1-0.3% sub-bar stack, not a lever; the kc_early+/edge_late- pair is the one worth a 32-rank arm if lanes are free. The bill route (handoff top block) remains the J2W path.
    JOINT FD DONE: both requested analyses plus clean helper ran. Correct paired6 B1 +0.02763% SE.17167, B2+.11878 SE.23085; regime C/flat. Old v5 secondnet warm-bill bias and no-op GSK/GK4 diagnosed. Bounded edge10 candidate built via make_joint_candidate; fresh validation running, provisional first3+.077%. Existing GSK/GK4 fresh settings now cache-live. No oracle-parabola gain claimed. Use eval_fresh_arm_v6/v7 (official0.12.3, per-netwarm), compare_relative_v1.py, summarize_fresh_v1.py.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/agent_ideas_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/mean_origin_v1.py](code/mean_origin_v1.py)
- [code/run_j5_v1.sh](code/run_j5_v1.sh)
- [code/candidate_dmt_sci_v1.py](code/candidate_dmt_sci_v1.py)
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

- [TEAM_LOG_2026-10-01_to_10-22.md#L6222](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6222)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6229](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6229)

<details>
<summary>All result files and records (8)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [joint_fd_settings_v1.json](joint_fd_settings_v1.json) · result
- [joint_fd_labels_v1.json](joint_fd_labels_v1.json) · result
- [IDEAS_v1.md](../../join-sketch/per-row-join-frame-energy-repair-increases-mse-0/IDEAS_v1.md) · record
- [IDEAS_v2.md](../../join-sketch/per-row-join-frame-energy-repair-increases-mse-0/IDEAS_v2.md) · record
- [joint_step_settings_v1.json](../../join-sketch/per-row-join-frame-energy-repair-increases-mse-0/joint_step_settings_v1.json) · result
- [IDEAS_v3.md](../../join-sketch/per-row-join-frame-energy-repair-increases-mse-0/IDEAS_v3.md) · record
- [census_a7ffd713_weighted_v1.json](../../join-sketch/per-row-join-frame-energy-repair-increases-mse-0/census_a7ffd713_weighted_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-joint-early-covariance-profile-eight-scalar-corrections-corrected-six) · [Copy provenance](../../../PROVENANCE.json)

