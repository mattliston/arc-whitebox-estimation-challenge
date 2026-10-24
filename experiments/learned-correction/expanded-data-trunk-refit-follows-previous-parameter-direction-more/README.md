# Expanded-data trunk refit follows the previous parameter direction more slowly

2026-09-24T19:18Z · experiment · informational · measured

Half an epoch on the larger training split matched the earlier first-epoch solution and remained worse than its second epoch. Parameter and per-layer effects indicated similar learning directions rather than new signal.

Recorded measurement:

    - 2026-09-24 12:21 analyst: T640 TRUNK MID-EVAL (update 80/161): VALID16 1.751280e-8 = -1.28% vs before (14/16); vs B2 ep0 (-1.31%) +0.03% (T640 better 10/16); vs B2 ep1 (1.74614e-8, -1.57%) +0.29% (4/16). Half an epoch of TRAIN640 at lr 5e-5 ~= B2 ep0; the second half (cosine decay) decides. params_mid80.pt saved.
    - 2026-09-24 12:18 lead: T640 trunk mid-eval (update 80/161): VALID16 1.75128e-8, a tie with B2 ep0 (+0.03%) and +0.29% vs B2 ep1 (4/16). It does NOT enter S12; the epoch finishes ~18:00, then VALID16/VALID40. Dead units in non-column work (analyst §29): with (a), (ii) covariance off-diagonals +1.80%, (iii) pair/GNN +1.79%, all together +2.74%: CLOSED, since they shift what the learned parts see for every unit. (i) star is +0.30% over (a), PARKED until column keying lands and its native saving can be priced after the batching penalty. The T640 knobs refit is RUNNING (resumed 11:56).
    - 2026-09-24 12:32 analyst: T640 trunk mid80 vs B2 ep1: SAME PATH, SLOWER. Param moves align (trunk cos .84, deep heads .81-.93; displacement .77-.97 of B2's). Per-layer mean-error gains cover 76/88/92/82% of B2 ep1's at L5-9/L10-12/L13-14/L15; the only difference is L1-3 (B2 +0.3..0.7% worse, T640 flat). Watchers armed for knobs mid (~13:25) and both ends (~17:50 trunk, ~19:05 knobs), compare_t640.py. RESULT.md sec 30.
    - 2026-09-24 12:32 lead: analyst §30: T640 mid-epoch is converging SLOWER ALONG B2's PATH, not learning something different. Parameter-move cosine vs B2 ep1 is .84 on the trunk and .71 on the heads; function gains are 76-92% of B2's per layer band. The only difference is that L1-3 stay flat where B2 made them slightly worse. Reading: the learned parts are NOT data-limited, since 4x data gives the same direction, so more data for the same architecture is not a big lever (relevant to the MPS question). End of T640 ~17:50 (trunk), knobs ~19:05.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_analyst_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/compare_t640.py](code/compare_t640.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L380](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L380)
- [TEAM_LOG_2026-09-22_to_09-26.md#L382](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L382)
- [TEAM_LOG_2026-09-22_to_09-26.md#L387](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L387)
- [TEAM_LOG_2026-09-22_to_09-26.md#L388](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L388)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-24-expanded-data-trunk-refit-follows-previous-parameter-direction-more) · [Copy provenance](../../../PROVENANCE.json)

