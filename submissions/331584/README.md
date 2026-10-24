# Submission #331584: exact-cost Winograd depth selection and a scalar final readout on the wall-repaired estimator

2026-09-19T17:44:39Z · submission · adopted · measured

Relative to #330931 (the wall-repaired shared rank-384 card with the I95 size rule) this file keeps the physics and learned payload byte-identical and changes two things: the Winograd depth of every product is selected by an exact kernel FLOP count plus a 1.5 MFLOP penalty per remote request, and the final-layer head is replaced by a scalar readout distilled from the head's own predictions. It graded as the new official best but stayed above the 4e-9 milestone; the local 14-net estimate 3.9806e-9 was optimistic by about .9%.

Recorded measurement:

    official adjusted 4.015535762022727e-9, raw 2.1687335731712666e-8, +4.12626% over #330931 (4.188358306539703e-9); accepted 2026-09-19T17:44:39.051Z; local NC14 3.9806136484e-9, projection 3.9960089480e-9; candidate SHA f06314111d7f41a7c1fbcb8d888a509d0e46222cd31b5ab227ec1e334e582a1f, archive 6356d5d81d14aced04611c5279f44e70f955abd2dbabe3d0f073ecc5132702e2 (770155 bytes)

Public score: 4.015535762022727e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [RECEIPTS.md#L167](../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L167)
- [RECEIPTS.md#L171](../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L171)
- [RECEIPTS.md#L206](../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L206)
- [RECEIPTS.md#L208](../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L208)
- [official_best_331584.json#L2](official_best_331584.json#L2)

<details>
<summary>All result files and records (7)</summary>

- [cost150_submission.json](cost150_submission.json) · record
- [official_best_331584.json](official_best_331584.json) · record
- [sub331584_poll_07.json](sub331584_poll_07.json) · record
- [cost150_panel_score.json](../../experiments/evaluation-methodology/exact-cost-depth-selection-passes-14-network-panel-adjusted/cost150_panel_score.json) · result
- [cost150_package.json](../../experiments/evaluation-methodology/exact-cost-depth-selection-passes-14-network-panel-adjusted/cost150_package.json) · record
- [upload_approvals_20260919_v1.json](upload_approvals_20260919_v1.json) · record
- [RECEIPTS.md](../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

## Related work

- ← shipped in: [Exact-cost depth selection passes 14-network panel: adjusted ratio .954075238, score 3.9806136484e-9](../../experiments/evaluation-methodology/exact-cost-depth-selection-passes-14-network-panel-adjusted/README.md)
- ← supersedes: [Exact-cost depth selector with a 750k instead of 1.5M FLOP request penalty: 14-net adjusted ratio .9906359025, held](../../experiments/arithmetic-kernels/exact-cost-depth-selector-750k-instead-1-5m-flop/README.md)
- ← shipped in: [Exact-cost Winograd depth selection plus 1.5 MF per request improves adjusted error: ratio .957179078](../../experiments/arithmetic-kernels/exact-cost-winograd-depth-selection-plus-1-5-mf/README.md)
- supersedes: [Submission #330931: batched six-level products, split frame caches and size-based depth selection passed](../330931/README.md)

[Research graph](../../graph/index.html#node=submission-331584) · [Copy provenance](../../PROVENANCE.json)

