# Submission #332492: approximate join core, shared star diagonal and refitted learned correction

2026-09-26T18:11:24Z · submission · adopted · measured

Submission #332492 used approximate join core, shared star diagonal and refitted learned correction. It graded successfully on all 100 public networks and became the best public adjusted score at that time.

Recorded measurement:

    Official adjusted score 2.22016596877829e-09; utilization 0.13183073144399715; 100 networks, 0 failures. Official raw MSE 1.684322921136072e-08; adjusted improvement over #332361 2.7360001074986817%; telemetry {"completed":100,"failed":0,"max_predict_wall_s":83.70165311400069,"mean_predict_wall_s":77.96953341609003,"max_residual_wall_s":0.26327298900000073,"mean_residual_wall_s":0.21026106370999997}.

Public score: 2.22016596877829e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [official_bank_S30_v1.json#L3](official_bank_S30_v1.json#L3)
- [SUBMISSION_LEDGER.md#L897](../../records/SUBMISSION_LEDGER.md#L897)

<details>
<summary>All result files and records (10)</summary>

- [official_bank_S30_v1.json](official_bank_S30_v1.json) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [eval_332492.json](eval_332492.json) · result
- [eval_332492_20260926T182037Z.json](eval_332492_20260926T182037Z.json) · result
- [eval_332492_20260926T182435Z.json](eval_332492_20260926T182435Z.json) · result
- [eval_332492_20260926T182824Z.json](eval_332492_20260926T182824Z.json) · result
- [eval_332492_20260926T183301Z.json](eval_332492_20260926T183301Z.json) · result
- [eval_332492_20260926T183624Z.json](eval_332492_20260926T183624Z.json) · result
- [eval_332492_20260926T184258Z.json](eval_332492_20260926T184258Z.json) · result
- [eval_332492-a43888dc.json](eval_332492-a43888dc.json) · result

</details>

## Related work

- ← shipped in: [Combined join-core and star-diagonal changes: all 40 held-out networks improve adjusted score](../../experiments/evaluation-methodology/combined-join-core-star-diagonal-changes-all-40-held/README.md)
- ← shipped in: [Future-basis Nyström core approximation: native bill falls 0.7069% and 0.7196%](../../experiments/join-sketch/future-basis-nystrom-core-approximation-native-bill-falls-0/README.md)
- ← shipped in: [Refitted correction and final-layer pair-mean truncation: held-out adjusted score improves 2.3550%](../../experiments/learned-correction/refitted-correction-final-layer-pair-mean-truncation-held-out/README.md)
- ← shipped in: [Star diagonal reuses an existing transport: native bill falls another 0.3372% and 0.3393%](../../experiments/arithmetic-kernels/star-diagonal-reuses-existing-transport-native-bill-falls-another/README.md)
- ← shipped in: [Unused terminal join-core update removed: all 16 layers remain bit-identical](../../experiments/flop-accounting/unused-terminal-join-core-update-removed-all-16-layers/README.md)

[Research graph](../../graph/index.html#node=submission-332492) · [Copy provenance](../../PROVENANCE.json)

