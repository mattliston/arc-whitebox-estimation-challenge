# Rank-two young-read tail completion: bill -1.34%, MSE +0.68%, all-24 adjusted -0.67%

2026-10-15T04:41:26Z · experiment · adopted · measured

The young-read tail completion (RTT2) truncates the young plane read to a gate-sorted prefix and completes the dropped tail with a fixed rank-2 v/mu term, trading a 0.7% accuracy loss for a 1.3% bill cut. It passed the pilot, the full 24-network science and the seal, and after a request repair became #336178, the largest adjusted gain of the window.

Recorded measurement:

    pilot 8: adjusted -0.607559%, MSE +0.716259%, bill -1.314403% (7/8 lower); full 24: adjusted -0.672423%, OOS 8 -0.503079%, MSE +0.679364%, bill -1.342666% (21/24 lower); original seal 8 log net -0.202779% (SE 0.415326%); repaired seal 8 -0.302560% (SE 0.363265%)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_read_tail_rank2_native_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_rtt2_HELD_v1.py](code/fragment_rtt2_HELD_v1.py)
- [code/candidate_rtt2_HELD_v1.py](code/candidate_rtt2_HELD_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3317](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3317)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3327](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3327)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3330](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3330)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3336](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3336)

<details>
<summary>All result files and records (2)</summary>

- [DECISION_initial8_v1.json](DECISION_initial8_v1.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #336178: young-read rank-two tail completion with request repair on #336125, bill -1.34%, 1.49701e-9](../../../submissions/336178/README.md)
- ← enables: [Second orthogonal star direction: pilot-8 +0.11%; a rank-2 v/mu read tail recovers 61% of the omitted young-band energy](../../../records/research/2026-10-15-second-orthogonal-star-direction-pilot-8-0-11-rank/README.md)
- ← shipped in: [Submission #336178: young-read rank-two tail completion with request repair on #336125, bill -1.34%, 1.49701e-9](../../../submissions/336178/README.md)

[Research graph](../../../graph/index.html#node=2026-10-15-rank-two-young-read-tail-completion-bill-1-34) · [Copy provenance](../../../PROVENANCE.json)

