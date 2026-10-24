# Submission #331676: 65-literal final-head ridge refit of #331672 failed: two networks exceeded the 0.4 s residual cap

2026-09-20T18:38:57Z · submission · failed · measured

Only 64 final-head coefficients and one final bias of the #331672 source were refit by one ridge (1e-3) rule on a fixed 20-network TRAIN set weighted by complete measured cost; all five 16-fit/4-held folds improved and the separate 8-net confirmation passed its prospective gate. Accepted 2026-09-20T18:38:57.304Z under the standing upload approval, it was graded with cases 02 (jennifer-jordan) and 03 (brandon-chen) failing RESIDUAL_WALL_TIME_EXHAUSTED, 98 of 100 completed, and the zero-output failure penalties produced an aggregate of 0.029849614989534205. Local matched cold residuals of 0.303376075 s (parent 0.314119061 s) and an earlier 0.387908846 s cold boundary run against the 0.4 s cap gave inadequate assurance against platform variability.

Recorded measurement:

    official adjusted 0.029849614989534205, raw 0.029849630568143617; 98/100 completed (50 sealed), 2 RESIDUAL_WALL_TIME_EXHAUSTED; local five-fold pooled adjusted ratio 0.991161805173858 (CV parent 3.791043418994608e-9 vs candidate 3.75753743866317e-9); confirmation8 3.679619886257271e-9 vs parent 3.689980619531317e-9 ratio 0.997192198457844 (5 wins, worst 1.00521321142583); candidate raw 1.978848090501857e-8; candidate walls 40.927401/38.017763/38.956525 s; archive 779736 bytes sha256 3ac8426f653d73b1130b6946c8439f82d203e399fba6ba668e04c3852f33db50; estimator sha256 b190987ab0b034f5542c4157c123c81fefb6941146216ae3f18a52eff6e5b441

Public score: 0.029849614989534205. Status: GRADED. Failed networks: 2.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [OFFICIAL_RESULT.md#L3](OFFICIAL_RESULT.md#L3)
- [OFFICIAL_RESULT.md#L4](OFFICIAL_RESULT.md#L4)
- [OFFICIAL_RESULT.md#L5](OFFICIAL_RESULT.md#L5)
- [OFFICIAL_RESULT.md#L11](OFFICIAL_RESULT.md#L11)
- [OFFICIAL_RESULT.md#L12](OFFICIAL_RESULT.md#L12)
- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L14](RESULT.md#L14)
- [RESULT.md#L19](RESULT.md#L19)
- [RESULT.md#L26](RESULT.md#L26)
- [RESULT.md#L46](RESULT.md#L46)
- [REEXAMINATION_INDEX_20260919.md#L882](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L882)
- [REEXAMINATION_INDEX_20260919.md#L884](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L884)
- [REEXAMINATION_INDEX_20260919.md#L872](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L872)

<details>
<summary>All result files and records (7)</summary>

- [OFFICIAL_RESULT.md](OFFICIAL_RESULT.md) · record
- [RESULT.md](RESULT.md) · record
- [official_result_331676.json](official_result_331676.json) · record
- [cv_result.json](cv_result.json) · result
- [closeout_official_v1.json](closeout_official_v1.json) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [REEXAMINATION_INDEX_20260919.md](../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- ← enables: [Standing approval for all qualified Phase 2 uploads recorded 2026-09-20 after the exact-archive approval of #331672](../../records/research/2026-09-20-standing-approval-all-qualified-phase-2-uploads-recorded-2026/README.md)
- ← shipped in: [Standing approval for all qualified Phase 2 uploads recorded 2026-09-20 after the exact-archive approval of #331672](../../records/research/2026-09-20-standing-approval-all-qualified-phase-2-uploads-recorded-2026/README.md)
- supersedes: [Submission #331672: full lower-moment support (full covariance, star and trace reads) added to the lazy-rotation parent](../331672/README.md)
- enables: [Submission #331682: lossless static parameter packing of the 65-coefficient final-head refit; 100/100 completed](../331682/README.md)
- ← supersedes: [Submission #331682: lossless static parameter packing of the 65-coefficient final-head refit; 100/100 completed](../331682/README.md)

[Research graph](../../graph/index.html#node=submission-331676) · [Copy provenance](../../PROVENANCE.json)

