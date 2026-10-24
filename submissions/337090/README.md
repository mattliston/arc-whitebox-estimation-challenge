# Submission #337090: final-layer third-cumulant row threshold 3.0 and own prebinding list

2026-10-19T10:16:36Z · submission · adopted · measured

The held refit base 5d097ef1 plus _L15_ROWCUT_Z 3.0 graded VALID 100/100 at 1.434644350e-9, the third bank step (< 1.435e-9): util -0.40% vs #337065 (local -0.295%: a final-layer cut transfers in full and more because the public networks keep more rows above z than the store networks), MSE +0.095% paired on the 50 public networks; the cut also trimmed the slowest networks (residual max .3972 -> .38, wall 117.6 -> 113.5 s).

Recorded measurement:

    exact 1.434644350e-9 = -0.304% vs #337065; util .108910 (-0.40%); MSE 1.317e-8 (+0.095% paired, se .054); residual public max .38; predict wall max 113.5 s; gates setup5 3.34 s, stage A 23 signatures; to J2W -1.03%, to oliver -1.14%, to 1.41e-9 -1.72%

Public score: 1.434644350303316e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L5932](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5932)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5936](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5936)
- [TEAM_LOG_2026-10-01_to_10-22.md#L5940](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L5940)
- [SUBMISSION_LEDGER.md#L1106](../../records/SUBMISSION_LEDGER.md#L1106)
- [SUBMISSION_LEDGER.md#L1107](../../records/SUBMISSION_LEDGER.md#L1107)

<details>
<summary>All result files and records (6)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [eval_337090.json](eval_337090.json) · result
- [setup5.log](setup5.log) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Final-layer third-cumulant row threshold reduced from 3.5 to 3.0: -0.177% adjusted on 32 networks](../../experiments/late-layer-exactness/final-layer-third-cumulant-row-threshold-reduced-3-5/README.md)
- supersedes: [Submission #337065: join sketch columns 1536/1024/1024 on #337061, graded 1.4390249e-9 (-0.252%, full bill transfer)](../337065/README.md)
- ← supersedes: [Submission #337126: layer-14 old-tail and young-crop exactness z 3.0 on #337090, graded 1.4332116e-9 (-0.100%)](../337126/README.md)

[Research graph](../../graph/index.html#node=submission-337090) · [Copy provenance](../../PROVENANCE.json)

