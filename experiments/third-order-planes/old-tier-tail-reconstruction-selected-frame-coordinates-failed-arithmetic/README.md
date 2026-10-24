# Old-tier tail reconstruction on selected frame coordinates failed the arithmetic cost test

2026-10-16T15:25:46Z · experiment · rejected · derived

A cheaper form of the old-tier tail band (rows with pre-activation ratio a in (2.5, 3.0] kept but reconstructed against only the top-k frame coordinates ranked by energy) was built in three arms. The v2 arms cost +7.8% / +6.8% bill because the core block was not 64-rounded (slow kernel route), not a physics result. The v3 arm with top-64 coordinates still cost +0.454% on seed 1 against +0.51% for the full band, so the band bill is the push of the extra kept rows through the weight matrix (n x cols per row, n = 1024 above rank 384), and a cheaper reconstruction cannot make it cheap.

Recorded measurement:

    OTTB v2 arms (ottbfull / ottb64 / ottb32): bill +7.8% / +6.8% on seed 1. OTTB64v3 (e7875613): bill +0.454% on seed 1 vs OTTP30 +0.51% for the full band.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_ladder2h_v1/build_ottb_v3.py](code/opus_ladder2h_v1/build_ottb_v3.py)
- [code/opus_ladder2h_v1/cand_s2tr_l2h_ottb64v3_v1.py](code/opus_ladder2h_v1/cand_s2tr_l2h_ottb64v3_v1.py)
- [code/opus_ladder2h_v1/cand_s2tr_l2h_ottb64_v3.py](code/opus_ladder2h_v1/cand_s2tr_l2h_ottb64_v3.py)
- [code/opus_s2tr_v1/candidate_S2tr_v1.py](code/opus_s2tr_v1/candidate_S2tr_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4267](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4267)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4276](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4276)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4290](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4290)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Old-tier tail z by side: positive side 3.0 buys -0.50% MSE for +0.51% bill; negative side and deeper cuts lose](../old-tier-tail-z-side-positive-side-3-0/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-old-tier-tail-reconstruction-selected-frame-coordinates-failed-arithmetic) · [Copy provenance](../../../PROVENANCE.json)

