# Exact select rosters with dense remainders: bill -0.47% warm but MSE +2.39%; the padding units carry 2.3% of accuracy

2026-10-16T22:19:33Z · experiment · rejected · measured

Roster rounding (select rosters to 32, dead-cut rosters to 64) was priced as the one remaining bill lever over 1%: the production kernel curve prices only 32/64-divisible dimensions (anything else runs dense at 1.0), rounding select up to 64 costs +2.81% bill for +0.03% MSE (so the exact-select ceiling is about -1.9%), rounding down is catastrophic, and the cheapest partial (dead cut 64 to 32) is negative. The dense-remainder family (exact rosters, aligned part through the trees plus leftover rows dense, Smirnov operands zero-padded at entry) was built on the estimator submitted as #336412 (S2tr): the exact response-plus-births select saved 0.47% warm bill but cost 2.39% MSE, additive over response-only (+0.69%) and births-only (+1.55%). The roughly 15 units per roster that rounding adds just below the 0.01 floor are a tuned accuracy budget, so exactness is a floor raise; the lever was declared dead.

Recorded measurement:

    Kernel curve: non-32 dims dense 1.0; 64-aligned 0.448-0.514 of dense; odd-32 +0.8-5%/element. Census (7103256): select pad 12.7 rows avg; dead-cut-type 29-42. sel64: bill +2.81% warm / +3.94% first, requests -1,210 / -882, client instructions -12.75%, MSE +0.03%; sel64d: bill -8.0%, MSE +48%; dc32 (003d0706): bill +0.26% warm / +0.17% first, MSE -0.02%, requests -98. rem1c (86eef776): bill +0.41% warm / +1.22% first, requests +436 / +1,054, MSE +2.28%; rem1d (b520f916, + Smirnov plan synthesis): bill -0.47% warm / +0.41% first, requests +517 / +1,197, instructions +2.3% / +4.5%, MSE +2.39%; rem1e response-only: bill +0.16%, requests +208, MSE +0.69%; rem1f births-only: bill -0.44% / +0.32%, requests +476 / +1,192, MSE +1.55%; sm40off diagnostic: bill +0.76%, requests -456, MSE +0.09%.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/lead/runs/opus_roster_v1/curve_v1.py](code/lead/runs/opus_roster_v1/curve_v1.py)
- [code/lead/runs/opus_roster_v1/census_nz_v1.py](code/lead/runs/opus_roster_v1/census_nz_v1.py)
- [code/lead/runs/opus_roster_v1/cand_rem1d.py](code/lead/runs/opus_roster_v1/cand_rem1d.py)
- [code/lead/runs/opus_roster_v1/cand_dc32.py](code/lead/runs/opus_roster_v1/cand_dc32.py)
- [code/lead/runs/opus_solo_v1/cand_sel64.py](code/lead/runs/opus_solo_v1/cand_sel64.py)
- [code/lead/runs/opus_solo_v1/cand_sel64d.py](code/lead/runs/opus_solo_v1/cand_sel64d.py)
- [code/analyst/fresh/loader.py](code/analyst/fresh/loader.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4504](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4504)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4510](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4510)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4512](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4512)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4515](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4515)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4517](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4517)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4518](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4518)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4520](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4520)

<details>
<summary>All result files and records (3)</summary>

- [RESULT.md](../../../records/research/2026-10-16-private-evaluation-retained-submission-336342-zero-failure-option-further/RESULT.md) · result
- [fragment_rem1d_v1.py.txt](fragment_rem1d_v1.py.txt) · code
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-exact-select-rosters-dense-remainders-bill-0-47-warm) · [Copy provenance](../../../PROVENANCE.json)

