# Submission #333559: norm-coupling law on column pairs folded into the third-order plane read

2026-10-02T14:52:36Z · submission · adopted · measured

candidate generation 40 (S40) is candidate generation 39 (S39) plus the lean law of the norm-coupling matrix carried as a second weight per bank column and a cross weight per column pair and folded into the (2,1) third-cumulant plane (K21) bank read, with three forms removed (the multiple of C from a per-layer table times the diagonal regression, two mean forms with the regression's own coefficients), the card's beta in the carried weights and all sources at strength 1. The lean port was built in ninety minutes from the lead's torch reference; qualified in a 30-minute quiet window; graded 1.7590e-9 and third place, Luna 4.6% behind. Its worst residual 0.365 s was a first network on a worker set up a second time 1.124x slow.

Recorded measurement:

    adjusted 1.7592335667503578e-9 (S39 1.8470e-9, -4.76%; expected about 1.73e-9); final-layer MSE 1.5070e-8 (-5.64%; native ten fresh -7.93%, torch 50 ranks -5.80%), all-layer 5.928e-9 (-6.6%), utilisation 0.11671985880106149 (+0.90%); 100/100, wall 79.83 / 91.18 s (projected 79.2 / 89.8), residual 0.311 / 0.365 s (projected 0.309 / 0.355); stage A five new einsum signatures; stage C requests 43,628 a network (S39 42,624), setup 3.43 s, bill 1.00876 x S39; stage D handles 57,810, footprint 2.69 GB, setup 4.08 s (limit 4.1); pair law 0.866% of the bill (fold 0.554%, forms 0.039%, update 0.273%), +8.4 G kernel instructions a network (put_along_axis exchange 5.18 G); worst network index 25: worker 3's second setup booked 29,381,392,716 flops = 627 of 649 keys bound after 16,902 requests (1.124x slow), first network at 1.138x median kernel time; fresh-run loss risk about 3% (readings 1.4-5.1%); third place (marius_binner 1.6407e-9, J2W 1.6161e-9 ahead by 6.7% and 8.1%)

Public score: 1.7592335667503578e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [../../experiments/fourth-order-terms/norm-coupling-law-column-pairs-lowers-mse-5-80/code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v4.py](../../experiments/fourth-order-terms/norm-coupling-law-column-pairs-lowers-mse-5-80/code/opus_reexam_20261001_v1/engineer/a38_pairlaw/fragment_np_v4.py)
- [unit_np3.py](unit_np3.py)

## Evidence

- [SUBMISSION_LEDGER.md#L915](../../records/SUBMISSION_LEDGER.md#L915)
- [SUBMISSION_LEDGER.md#L916](../../records/SUBMISSION_LEDGER.md#L916)
- [TEAM_LOG_2026-10-01_to_10-22.md#L334](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L334)
- [TEAM_LOG_2026-10-01_to_10-22.md#L344](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L344)
- [TEAM_LOG_2026-10-01_to_10-22.md#L345](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L345)
- [TEAM_LOG_2026-10-01_to_10-22.md#L347](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L347)
- [TEAM_LOG_2026-10-01_to_10-22.md#L348](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L348)
- [TEAM_LOG_2026-10-01_to_10-22.md#L350](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L350)
- [TEAM_LOG_2026-10-01_to_10-22.md#L353](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L353)
- [TEAM_LOG_2026-10-01_to_10-22.md#L357](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L357)
- [TEAM_LOG_2026-10-01_to_10-22.md#L367](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L367)
- [TEAM_LOG_2026-10-01_to_10-22.md#L373](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L373)
- [SESSION_NOTES_2026-10-01_to_10-03.md#L575](../../records/SESSION_NOTES_2026-10-01_to_10-03.md#L575)

<details>
<summary>All result files and records (10)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [attempt.json](attempt.json) · record
- [receipt.json](receipt.json) · record
- [eval_summary.txt](eval_summary.txt) · result
- [eval_333559.json](eval_333559.json) · result
- [s40_651.json](s40_651.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [eval_333559-1b5398ce.json](eval_333559-1b5398ce.json) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SESSION_NOTES_2026-10-01_to_10-03.md](../../records/SESSION_NOTES_2026-10-01_to_10-03.md) · record

</details>

## Related work

- ← shipped in: [Norm-coupling law on column pairs lowers MSE 5.80% on 50 ranks; native implementation enters submission #333559](../../experiments/fourth-order-terms/norm-coupling-law-column-pairs-lowers-mse-5-80/README.md)
- supersedes: [Submission #333528: joint dial solve and third-order remainder carried by row sums cut bill 2.2%](../333528/README.md)
- ← supersedes: [Submission #333587: young-pair fourth-chaos plane, norm-coupling dial solve and fewer kernel requests](../333587/README.md)

[Research graph](../../graph/index.html#node=submission-333559) · [Copy provenance](../../PROVENANCE.json)

