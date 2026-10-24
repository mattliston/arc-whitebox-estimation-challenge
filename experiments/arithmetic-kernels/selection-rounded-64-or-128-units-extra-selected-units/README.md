# Selection rounded to 64 or 128 units: extra selected units outweigh deeper multiplication

2026-09-23T03:10Z · experiment · rejected · measured

Rounding selected row counts enabled deeper multiplication but introduced extra units and worsened the metered error tradeoff.

Recorded measurement:

    - 2026-09-22 20:10 [engineer] follow-ups: select() rounding to 64/128 measured on the bank (R64: util r35 +1.06%/r100 -0.51%, MSE r35 -2.5%/r100 +5.5%; R128 util +3.6%/+3.5%, MSE +5.8%/+29%): extra units cost more than the depth they enable; only 3 pushes on r35 were depth-capped by /64. Testing zero-padding those pushes to /64 instead (push64, physics unchanged). Frame bill over 12 nets in one process: 4.1% first net, ~1.1% steady state (not decaying).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fable_engineer_v1/batch_fragment.py](code/fable_engineer_v1/batch_fragment.py)
- [code/fable_engineer_v1/bigarena_fragment.py](code/fable_engineer_v1/bigarena_fragment.py)
- [code/fable_engineer_v1/bindhold_fragment.py](code/fable_engineer_v1/bindhold_fragment.py)
- [code/fable_engineer_v1/build_eng.py](code/fable_engineer_v1/build_eng.py)
- [code/fable_engineer_v1/build_f32.py](code/fable_engineer_v1/build_f32.py)
- [code/fable_engineer_v1/caps_fragment.py](code/fable_engineer_v1/caps_fragment.py)
- [code/fable_engineer_v1/caps_fragment_v3.py](code/fable_engineer_v1/caps_fragment_v3.py)
- [code/fable_engineer_v1/coldbind_fragment.py](code/fable_engineer_v1/coldbind_fragment.py)
- [code/fable_engineer_v1/evict_fragment_all.py](code/fable_engineer_v1/evict_fragment_all.py)
- [code/fable_engineer_v1/evict_fragment_lru.py](code/fable_engineer_v1/evict_fragment_lru.py)
- [code/fable_engineer_v1/evict_fragment_push5.py](code/fable_engineer_v1/evict_fragment_push5.py)
- [code/fable_engineer_v1/evict_fragment_push5r.py](code/fable_engineer_v1/evict_fragment_push5r.py)

<details>
<summary>Additional programs and dependencies (21)</summary>

- [code/fable_engineer_v1/grouped_fragment_v5.py](code/fable_engineer_v1/grouped_fragment_v5.py)
- [code/fable_engineer_v1/grouped_fragment_v7.py](code/fable_engineer_v1/grouped_fragment_v7.py)
- [code/fable_engineer_v1/grouped_fragment_v7b.py](code/fable_engineer_v1/grouped_fragment_v7b.py)
- [code/fable_engineer_v1/grouped_fragment_v7c.py](code/fable_engineer_v1/grouped_fragment_v7c.py)
- [code/fable_engineer_v1/grouped_fragment_v9.py](code/fable_engineer_v1/grouped_fragment_v9.py)
- [code/fable_engineer_v1/grouped_fragment_v9d.py](code/fable_engineer_v1/grouped_fragment_v9d.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v5.py](code/fable_engineer_v1/grouped_read5_fragment_v5.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v7.py](code/fable_engineer_v1/grouped_read5_fragment_v7.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v7b.py](code/fable_engineer_v1/grouped_read5_fragment_v7b.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v7c.py](code/fable_engineer_v1/grouped_read5_fragment_v7c.py)
- [code/fable_engineer_v1/grouped_read5_fragment_v9.py](code/fable_engineer_v1/grouped_read5_fragment_v9.py)
- [code/fable_engineer_v1/holdelim_fragment.py](code/fable_engineer_v1/holdelim_fragment.py)
- [code/fable_engineer_v1/holdonly_fragment.py](code/fable_engineer_v1/holdonly_fragment.py)
- [code/fable_engineer_v1/lean_fragment_v3.py](code/fable_engineer_v1/lean_fragment_v3.py)
- [code/fable_engineer_v1/push64_fragment.py](code/fable_engineer_v1/push64_fragment.py)
- [code/fable_engineer_v1/select_round_fragment.py](code/fable_engineer_v1/select_round_fragment.py)
- [code/fable_engineer_v1/setupbind_fragment_v3.py](code/fable_engineer_v1/setupbind_fragment_v3.py)
- [code/fable_engineer_v1/sq6_fragment.py](code/fable_engineer_v1/sq6_fragment.py)
- [code/fable_engineer_v1/symout_fragment_v3.py](code/fable_engineer_v1/symout_fragment_v3.py)
- [code/fable_engineer_v1/usedslots_fragment.py](code/fable_engineer_v1/usedslots_fragment.py)
- [code/fable_refit_v1/score_variant.py](code/fable_refit_v1/score_variant.py)

</details>

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L25](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L25)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-selection-rounded-64-or-128-units-extra-selected-units) · [Copy provenance](../../../PROVENANCE.json)

