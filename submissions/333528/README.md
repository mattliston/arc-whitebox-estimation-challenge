# Submission #333528: joint dial solve and third-order remainder carried by row sums cut bill 2.2%

2026-10-02T11:39:02Z · submission · adopted · measured

candidate generation 39 (S39) is the candidate generation 38 (S38) stack rebuilt on the candidate 3 (exact S2 term of the hub-coefficient mixture at 0.444 / 0.171 by age, kurtosis term 0.86, correction x 0.784, star leg j2 x 0.791, edge x 0.75) with the exact-term module in place of the column shift and the third-order remainder plane carried by its row sums alone. Qualified in a 26-minute quiet window with legs S38 / file / file / S38; graded 1.8470e-9, 4th place 0.18% behind Luna; the ten-rank native estimate of the error change had been -0.04% (se 0.39) where the board gave -0.93%: ten ranks size an error change only to about +-1%. It passed the designation test with the narrowest margins so far (mean residual 0.003 s under the rule).

Recorded measurement:

    adjusted 1.8467751795899251e-9 (S38 1.9070e-9, -3.15%; expected about 1.864e-9); final-layer MSE 1.5970e-8 (-0.93%), all-layer 6.344e-9, utilisation 0.11567516838750634 (-2.22%); 100/100, wall 77.29 / 87.62 s (projected 77.6 / 88.9), residual 0.302 / 0.347 s (projected 0.300 / 0.338); stage C requests 42,624 a network, bill 0.97769 x S38, kernel 1.0045 x S38; stage D handles 57,755, footprint 2.47 GB (RSS 7.23 GB); native ten fresh vs S38: MSE -0.04% [-0.75,+0.67], bill -2.239%, adjusted -2.27%, +659 calls (module +809, star -150); projection check: wall +0.3 s, kernel +0.5%, residual -0.7%, residual per request 7.08 microseconds, seven setups all complete; designation: room 0.098 s, slow setup +1.79%, first loss at 1.13x residual / 1.37x wall; fresh-run loss risk about 2% (1.0% by the first-network model)

Public score: 1.8467751795899251e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L913](../../records/SUBMISSION_LEDGER.md#L913)
- [SUBMISSION_LEDGER.md#L914](../../records/SUBMISSION_LEDGER.md#L914)
- [TEAM_LOG_2026-10-01_to_10-22.md#L256](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L256)
- [TEAM_LOG_2026-10-01_to_10-22.md#L260](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L260)
- [TEAM_LOG_2026-10-01_to_10-22.md#L264](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L264)
- [TEAM_LOG_2026-10-01_to_10-22.md#L265](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L265)
- [TEAM_LOG_2026-10-01_to_10-22.md#L266](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L266)
- [TEAM_LOG_2026-10-01_to_10-22.md#L269](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L269)
- [TEAM_LOG_2026-10-01_to_10-22.md#L271](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L271)
- [TEAM_LOG_2026-10-01_to_10-22.md#L275](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L275)
- [TEAM_LOG_2026-10-01_to_10-22.md#L276](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L276)
- [TEAM_LOG_2026-10-01_to_10-22.md#L277](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L277)
- [TEAM_LOG_2026-10-01_to_10-22.md#L290](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L290)
- [TEAM_LOG_2026-10-01_to_10-22.md#L292](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L292)
- [TEAM_LOG_2026-10-01_to_10-22.md#L303](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L303)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L526](../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L526)

<details>
<summary>All result files and records (13)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [attempt.json](attempt.json) · record
- [receipt.json](receipt.json) · record
- [eval_summary.txt](eval_summary.txt) · result
- [eval_333528.json](eval_333528.json) · result
- [lead_s39_chain.log](lead_s39_chain.log) · log
- [stage_c.json](stage_c.json) · result
- [quick_s39_cand3_five_ss0_fresh_a.json](quick_s39_cand3_five_ss0_fresh_a.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [eval_333528-e549d259.json](eval_333528-e549d259.json) · result
- [eval_333528-b8701718.json](eval_333528-b8701718.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- ← shipped in: [Joint least-squares fit of five dials improves held-out error by 0.70%](../../experiments/dials/joint-least-squares-fit-five-dials-improves-held-out/README.md)
- ← shipped in: [Row-sum representation of the third-order remainder saves 2.37% work for 0.50% additional error](../../experiments/third-order-planes/row-sum-representation-third-order-remainder-saves-2-37/README.md)
- supersedes: [Submission #333481: fitted dials, masked covariance correction and three-request operand loads](../333481/README.md)
- ← supersedes: [Submission #333559: norm-coupling law on column pairs folded into the third-order plane read](../333559/README.md)

[Research graph](../../graph/index.html#node=submission-333528) · [Copy provenance](../../PROVENANCE.json)

