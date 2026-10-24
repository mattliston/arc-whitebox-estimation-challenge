# Fixed node-head fidelity: pass, not an accuracy result

2026-09-09 UTC (2026-09-08 Pacific). No parameter updates in these checks.

Tiny native/CPU/MPS judge passed, SHA256 `e255b12c838bec49726eeec5d5d8f94694d4ce9bb3b5099b4525d1645b1140ff`.

Full-width native/CPU/MPS judge passed, SHA256 `66ddcb87f1abeb45438da7598422aee373a376e66997e6a5a33dace07eb1f434`. Fixed development ranks 35/100; zero head and fixed nonzero prototype; all 16 post-ReLU means. Native parent/zero/nonzero each also include exact repeat35.

- A=0 reproduces all 16 native parent output bytes, on all three calls.
- Added paid head work: 14,853,888 FLOPs per warm call, plus 512 FLOPs cold parameter setup. This is measured prototype overhead, not a future fitted total bill.
- Maximum native/shadow mean difference: 2.86102294921875e-6, below the prospective 1e-5 gate.
- CPU/MPS fixed-cotangent A-gradient relative errors: rank35 0.0012942266263578383; rank100 0.0002691941419957581, below the 0.05 gate. Non-negligible CPU norms: 2.057291652745952e-5 and 1.9324722656663348e-5.
- Existing 1e9-truth final-MSE ratios: 0.9995490635595055–1.0004896947443824 across fixed prototypes and both shadows, within the prospective 1% gate. These truths serve numerical fidelity only, not an improvement verdict.
- CPU stage: 154.8219 seconds, peak RSS 8,915,927,040 bytes, below its 12GB cap.
- MPS stage: 25.3975 seconds, peak reported RSS 1,324,941,312 bytes; recorded stage driver usage below 16GB cap.

Frozen train.py SHA256 `f2982348c5a56676443c3ab2cc24370e14fcf1c06150ea2ff09d285d39a9949f` was fully reviewed by root and independently reviewed by fresh_accuracy, with no launch-blocking defect found. The prescribed sole-MPS fixed 20-TRAIN/4-VALID pilot is admitted. Only terminal A after six epochs/30 updates may be judged, under PROTOCOL.md's complete-cost-adjusted and joint-win criteria. No extension, tuning, refit or automatic larger run.

Phase 2 bank confirmation #330390 contains only the earlier fixed-coefficient d50 candidate, not this node-conditioned head.
