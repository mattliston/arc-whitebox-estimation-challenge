# Chained candidate launch requires unique outputs and correctly expanded argument arrays

2026-09-25T20:15Z · incident · informational · measured

A validation chain first collided with a preexisting scratch output and then passed all shell arguments as one word. Archiving the stale output and relaunching with correct argument expansion restores the chain.

Recorded measurement:

    2026-09-25 13:15 porter: S29 chain LAUNCHED at S28 staging (args stack/S29_args_launched.txt: fable_engineer_v1/ksplit_fragment_v5w.py 57656b02336ec2c7 | params fable_physicist_v1/refit6b/s6b_loopR7s_joint_3ep/params_epoch1.pt 0a8fbdad7733a076 | d7sq fable_engineer2_v1/depth7/d7sq_fragment.py 8b863bbd31e3e560 | extra keys none; log fire_chain_S29.out).
    2026-09-25 13:15 porter: S29 NOT staged: /Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1/stack/s29_l7_exclude.json already exists (never overwrite)
    - 2026-09-25 13:16 lead: S28 STAGED (porter, 13:15): fable_fire_v1/fire_candS28/candS28_S27_refit_ks_cap_sched_g5_mem.py sha256 a559b073... = S24 + j3cut + k-split v5w + cap 480 + loopR7 epoch 1 literals (28 bit-equal) + schedule 384/320/384 + group 5 v2 + leaf-buffers-first + guard 4.0 s + value-ordered 460-key list + treuse. Gates: VALID40 vs S27 adjusted -1.788% [-2.40, -1.19] 31/40; residual cold +0.006 s (rule +0.010); setup under concurrent load [3.71, 2.52, 2.96] s, all 484 keys bound (gate 4.1 s; S27 [2.80, 2.04, 2.24] with 394); einsum/validate/treuse identity PASS. FIRES at 17:00 PDT (00:00 UTC, fresh daily count); S29 follows once staged.
    2026-09-25 13:18 porter: S29 RELAUNCHED 13:17:51 (same args: loopR7s ep1 0a8fbdad/7a2f997e, v5w, cap480, 384/320/384, g5 b769aa28, d7cp v2 64210b64 + d7sq 8b863bbd, no extra keys, bar -0.3). First launch 13:15:56 died at the collision check on stack/s29_l7_exclude.json (left by my cancelled standalone census; archived as s29_l7_exclude_standalone_census_1248_unused.json); a zsh relaunch passed the args as one word and died before writing (log kept). Lesson in memory (chain-generation-pitfalls #7). S28 STAGED 13:15:55 (residual +0.006 s PASS; setup under load max 3.712 s, 484 bound).

## Programs

- [../../../experiments/setup-budget/value-ordered-prebinding-cache-cap-480-reduce-repeat-binds/code/fable_engineer_v1/ksplit_fragment_v5w.py](../../../experiments/setup-budget/value-ordered-prebinding-cache-cap-480-reduce-repeat-binds/code/fable_engineer_v1/ksplit_fragment_v5w.py)
- [../../../experiments/arithmetic-kernels/depth-seven-packed-square-products-combine-reduce-arithmetic/code/d7sq_fragment.py](../../../experiments/arithmetic-kernels/depth-seven-packed-square-products-combine-reduce-arithmetic/code/d7sq_fragment.py)
- [../../../submissions/332361/estimator.py](../../../submissions/332361/estimator.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1270](../../TEAM_LOG_2026-09-22_to_09-26.md#L1270)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1271](../../TEAM_LOG_2026-09-22_to_09-26.md#L1271)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1272](../../TEAM_LOG_2026-09-22_to_09-26.md#L1272)
- [TEAM_LOG_2026-09-22_to_09-26.md#L1273](../../TEAM_LOG_2026-09-22_to_09-26.md#L1273)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [S29_args_launched.txt](S29_args_launched.txt) · record
- [s29_l7_exclude.json](s29_l7_exclude.json) · result
- [s29_l7_exclude_standalone_census_1248_unused.json](s29_l7_exclude_standalone_census_1248_unused.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-chained-candidate-launch-requires-unique-outputs-correctly-expanded-argument) · [Copy provenance](../../../PROVENANCE.json)

