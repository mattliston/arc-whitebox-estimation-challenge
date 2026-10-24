# Joint least-squares fit of five dials improves held-out error by 0.70%

2026-10-02T10:19Z · experiment · adopted · measured

Three shrink directions from stored exact gradients (learned covariance edge x 0.75, star leg j2 x 0.9, mean correction x 0.75) gave candidate 3-prelim at -0.25% against candidate 2 (not measurable natively on ten ranks). A joint least-squares solve over 12 single changes on the exact-term card (each one forward run, the physicist's mixture method, 16 line-search ranks) gave -0.87% against candidate 2 on the 50 judging ranks; the five-change subset that needs no new port (exact S2 term 0.444 / 0.171 by age, kurtosis term 0.86, correction x 0.784, star leg j2 x 0.791, edge x 0.75) gave -0.70% and was built as the trained part of S39. The solve also asked for more kurtosis strength (0.75 -> 0.86).

Recorded measurement:

    candidate 3-prelim: -1.69% vs untrained (se 0.264, 40 of 50), -0.252% vs candidate 2 (se 0.089, 31 of 50); native ten fresh +0.02% [-0.41,+0.52] vs S38, bill identical; joint solve (12 changes): 1.5845e-8, -2.300% vs untrained (se 0.279, 43 of 50), -0.870% vs candidate 2 (se 0.187, 38 of 50); five changes: 1.5872e-8, -2.13% vs untrained (se 0.257, 45 of 50), -0.702% vs candidate 2 (se 0.152, 35 of 50), all layers -1.704%; closed by line search 3: final-layer k4 x 0.96 / 0.92 0.00% / +0.33%, s_carry depth profile -0.11% / -0.13%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/trainer`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidates/runM_dials3/candidate.py](code/candidates/runM_dials3/candidate.py)
- [code/candidates/runM_cand3_five/candidate.py](code/candidates/runM_cand3_five/candidate.py)
- [code/m_mix.py](code/m_mix.py)
- [code/m_shrink_table.py](code/m_shrink_table.py)
- [code/otr_common.py](code/otr_common.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L180](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L180)
- [TEAM_LOG_2026-10-01_to_10-22.md#L204](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L204)
- [TEAM_LOG_2026-10-01_to_10-22.md#L225](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L225)
- [TEAM_LOG_2026-10-01_to_10-22.md#L239](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L239)
- [TEAM_LOG_2026-10-01_to_10-22.md#L242](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L242)
- [TEAM_LOG_2026-10-01_to_10-22.md#L249](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L249)
- [RESULT.md#L607](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md#L607)
- [RESULT.md#L661](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md#L661)

<details>
<summary>All result files and records (4)</summary>

- [spec.json](spec.json) · config
- [RESULT.md](../../../records/research/2026-10-01-training-engine-reproduces-reference-losses-gradients-exported-bytes/RESULT.md) · result
- [quick_s39_dials3_fresh_a.json](quick_s39_dials3_fresh_a.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- shipped in: [Submission #333528: joint dial solve and third-order remainder carried by row sums cut bill 2.2%](../../../submissions/333528/README.md)

[Research graph](../../../graph/index.html#node=2026-10-02-joint-least-squares-fit-five-dials-improves-held-out) · [Copy provenance](../../../PROVENANCE.json)

