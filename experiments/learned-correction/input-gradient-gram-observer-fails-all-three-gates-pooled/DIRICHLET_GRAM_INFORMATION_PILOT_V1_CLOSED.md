# Dirichlet-Gram information pilot: qualified scientific failure

Phase2 only. Completed September6,2026 at18:39:49 PDT
(result timestamp2026-09-07T01:39:49.184655Z). The single prescribed
execution was unified session51356, terminal exit0. Do not resume/restart.

## Decisive result

Final-layer raw MSE against the existing1e9 reference means, using the
frozen scalar-final prefix and served max0->float32 outputs. TRAIN20 was
ranks35..58 excluding the four VALID ranks below. These are reused research
holdouts, not the independent14-net release panel or an official score.

| Rank | Parent bank | Same-size retrained control | Dirichlet sources | Sources / parent |
| --- | ---: | ---: | ---: | ---: |
| 37 | 2.1281208136151524e-8 | 2.1441189951375490e-8 | 2.1417878160587998e-8 | 1.00642209895 |
| 39 | 1.7861887365936750e-8 | 1.7964439490428052e-8 | 1.7772879041105547e-8 | 0.99501685779 |
| 42 | 2.3223941922589634e-8 | 2.3297527213591250e-8 | 2.3198925451780220e-8 | 0.99892281548 |
| 48 | 2.2439329423914813e-8 | 2.2450744517045512e-8 | 2.2495971026512830e-8 | 1.00252421102 |

Pooled raw means: parent2.120159171214818e-8,
control2.1288475293110075e-8, sources2.1221413419996646e-8.
Sources/parent=1.00093491602506, sources/control=0.99684985081411;
only ranks39 and42 win against the parent. All three fixed advancement
gates (.65/.70/at least3 wins) FAIL. The sources are0.09349% worse than the
parent, and0.31501% better than the retrained control: no useful raw gain
to pay for the extra full-width state. No adjusted score was measured.

## Why this is a scientific verdict, not a harness failure

- All24 input-only rows had passed and were sealed before TRAIN targets.
- Both134->128->128->1 arms completed40epochs/120updates with the fixed
  initialization, seed930080, identical40 minibatch-permutation hashes,
  TRAIN-only source statistics, AdamW schedule and final-epoch-only export.
- Both payloads were sealed at01:39:46.440549Z; all held-out predictions
  were sealed at01:39:47.852130Z; the first VALID truth access was later,
  at01:39:47.854142Z. No validation loss was used during fitting.
- Ordinary NumPy and MPS held-out deltas differ by at most
  2.9103830456733704e-11, below the fixed2e-9 tolerance. The parent,
  control and source served outputs all use the same max0/float32 transform.
- Source/input seals passed throughout. Total execution11.768510459s,
  peak process RSS465,371,136bytes, nice10, MPS fallback0 and two threads.
  No failure receipt exists. Independent agent recomputed the scalar
  ratios and audited source/metadata/seal chronology; it found no defect.

Root read the full trainer before execution and independently checked
the completed source/artifact hashes and seal chronology. The independent
auditor opened no arrays and performed no numerical replay. Neither this
test nor that audit proves native D4 feature parity, wide PSD, true deep
Jacobian fidelity, complete inference cost or grader runtime.

## Frozen ownership

- Trainer SHA256:45c0271382f48dfbeecce7107a96456e70a3fee3e74a829b9334a2416ab71733.
- Result `dirichlet_gram_probe_pilot_v1/result.json`:
  e0f70e06cd38d7dd3a82c00550b9123f3d14e9cf957f1d2edd942137559c5e31.
- Control payload:73e65bafa94d2dbc91644e47f8324313ebf5012a8755988d97cc939b1beeb074.
- Source payload:3e54c8ef02895fcf41e5076a08efcd7245c9bfe2a00c5ff41a980d0712af55c3.
- Held-out prediction archive:
  7ca41c656ec1f325dee4a1119b9648c076b738969b50282e44ede44b9b987777.
- Fixed-fit root acceptance:
  4796e8faa5d42766aaf6f8dd25315acfa442d95a3ab4e1cc9f9be09d3f7e5c3a.

The prescribed observer/features/order/window/fit is CLOSED. No added
channels, altered gate moment kernel, seeds, scales, extra epochs, capacity,
state writes,200-net expansion, native port, panel or upload follows.
This does not prove that every gradient-based method lacks information.
All artifacts are preserved; the usable bank remains#329926 at official
adjusted4.982231460183304e-9. The strict<1e-9 goal is not achieved.
