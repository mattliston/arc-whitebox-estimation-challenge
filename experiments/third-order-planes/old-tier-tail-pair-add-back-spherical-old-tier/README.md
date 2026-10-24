# Old-tier tail pair add-back and spherical old-tier tail: weighted error worse or +0.42% adjusted on the first network

2026-10-17T03:09:54Z · experiment · rejected · measured

Two old-tier tail (OTT) variants were closed: adding back exact pair contributions of 128/256 signed pairs ranked by rho^3 times the exact pair tensor energy recovered raw cubic error but made the mean-weighted error worse (-5.99% / +2.35%), at +.208/.416% bill in an 8-call scenario; a fixed spherical OTT (669f300d) read +0.41995% adjusted on the first paired network and +0.02533% mean adjusted on 4, so both were closed.

Recorded measurement:

    Pair add-back saved 651 L13: rounded-tail raw cubic error^2 recovery 8.22/16.37%, mean-weighted -5.99%/+2.35%, 2.5-3 band weighted -3.62/+6.29%; stable CD cubic FP32 ~9e-7 relative; 8 equal L13 calls ~.208/.416% bill. Sphere OTT: saved 651 diagnostic 12.63% weighted component error recovery; paired 378 adjusted +0.41995%; fixed 4 mean adjusted +.02533%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/astra_crunch_ott_pairs_v1/saved_v1.py](code/lead/runs/astra_crunch_ott_pairs_v1/saved_v1.py)
- [code/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py](code/lead/runs/astra_crunch_physics_v1/guard_memory50_v1.py)
- [code/lead/runs/astra_crunch_physics_v1/ott_spherical_v1/aggregate_v1.py](code/lead/runs/astra_crunch_physics_v1/ott_spherical_v1/aggregate_v1.py)
- [code/lead/runs/astra_crunch_physics_v1/ott_spherical_v1/candidate_v17_sphere_v1.py](code/lead/runs/astra_crunch_physics_v1/ott_spherical_v1/candidate_v17_sphere_v1.py)
- [code/lead/runs/astra_crunch_physics_v1/ott_spherical_v1/judge_v1.py](code/lead/runs/astra_crunch_physics_v1/ott_spherical_v1/judge_v1.py)
- [code/lead/runs/astra_crunch_physics_v1/ott_spherical_v1/run_one_v1.py](code/lead/runs/astra_crunch_physics_v1/ott_spherical_v1/run_one_v1.py)
- [code/lead/runs/astra_crunch_physics_v1/ott_spherical_v1/saved_v2.py](code/lead/runs/astra_crunch_physics_v1/ott_spherical_v1/saved_v2.py)
- [code/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v17.py](code/lead/runs/astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v17.py)
- [code/release/rusage.py](code/release/rusage.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4762](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4762)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4766](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4766)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4772](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4772)

<details>
<summary>All result files and records (2)</summary>

- [CLOSURE_v1.json](CLOSURE_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-17-old-tier-tail-pair-add-back-spherical-old-tier) · [Copy provenance](../../../PROVENANCE.json)

