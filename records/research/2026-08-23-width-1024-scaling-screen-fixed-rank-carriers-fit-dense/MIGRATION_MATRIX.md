# Phase 1 -> Phase 2 migration matrix

| Research component | Phase 2 disposition | Reason / first gate |
|---|---|---|
| Universal cumulant and ReLU identities | Reuse | Shape-independent mathematics; revalidate numerical spelling only. |
| CP3 carrier and K21 overlay | Port, then re-meter | `O(depth*width^2*rank)` geometry tracks the 8x budget increase closely. |
| Centered TT3 residual | Port mechanism, reselect rank | Rank 22, sketches, rounding thresholds, and CAP3 schedule are Phase 1 empirical choices. |
| Dense covariance/PSD Cholesky certificate | Redesign first | Fifteen naive width-1024 repairs screen at 323.1B FLOPs before the rest of the estimator. |
| Correlation-interior PSD rule | High-priority port | It avoids a separate dense repair, but must be expressed and metered in flopscope 0.12. |
| Local Edgeworth mean link | Reuse formula, recalibrate inputs | Wider/shallower geometry should improve the expansion; all empirical clipping/scales are stale. |
| Learned diagonal moment transport | Retrain from scratch | Width, depth, layer index, feature distribution, and error law changed. |
| 32-channel W/W2/W3/W4 observer | Conditional Phase 2 lane | The shared per-neuron form is width-equivariant, but its checkpoint, layer scales, and 32-step recurrence are not. Re-meter powers under 0.12 and consider fewer depth-specific channels. |
| A160 teacher feedback result | Capacity clue only | It proves a coherent mean/variance/K3 interface can cross 1e-7 at 256x32, not that the Phase 2 provider does. Re-run the substitution ladder at 1024x16. |
| A161-A166 learned-feedback ladder | Archive | It localizes closed-loop drift in Phase 1. A166 is suspended and has no disk checkpoint; do not resume it as a Phase 2 run. |
| Phase 1 raw MSE curves and official best | Archive unchanged | Layer 32 and the old networks do not exist in Phase 2. Never relabel these curves. |
| Phase 1 FLOP receipts | Invalid for Phase 2 | Shape and flopscope pricing both changed. The padded >52-label underbilling path is closed. |
| Phase 1 public/train splits | Invalid for Phase 2 | The v2 mini/full networks and seeds are disjoint new cohorts. |
| Sampling / whitening / QMC | Fresh comparator lane | Brute-force sample capacity stays near 65k; public reports make whitening a strong hybrid comparator. |
| Offline MPS acceleration | Allowed for research | Use for data extraction and training after CPU/MPS parity gates; it is unavailable in submitted `predict()`. |

## External evidence that changes the ordering

The comments on AIcrowd topic 18177 report a zero-fit second-order
Hermite/Mehler chain versus FLOP-matched Monte Carlo across widths and depths.
At the Phase 1 shape (256x32), the analytic/MC MSE ratio was about 1.04.  At
1024x16 it was 0.12.  The complete reported depth-16 row was 0.64, 0.83,
0.37, 0.21, 0.12 as width increased 64, 128, 256, 512, 1024.  This directly
supports running the analytic hierarchy before a learned observer in Phase 2.

This is participant evidence, not an organizer guarantee.  Our graph records
it as a prioritization input and requires an independent v2-phase2 replay.

## Phase 2 research lanes

1. **Authority and reproducibility.** Pin dataset, versions, shapes, rules,
   resource limits, and exact mini/full separation.
2. **Meter rebasing.** Run every primitive through the 0.12 client/server and
   subprocess harness; remove Phase 1 pricing tricks and cubic repairs.
3. **Analytic hierarchy.** Measure diagonal Gaussian, second-order
   Hermite/Mehler, CP3, then CP3+TT3 under one frozen evaluation protocol.
4. **Bottleneck attribution.** Substitute teacher mean, variance, K3, and K4
   on Phase 2 mini to determine what still limits layer-16 mean.
5. **Learned correction.** Only if a residual remains, retrain a shallower
   recurrent observer end to end on Phase 2 states with autonomous selection.
6. **Sampling comparator.** Meter whitening/QMC/control-variate baselines and
   hybridize only when they beat the analytic lane at equal official cost.
7. **Promotion.** Seal target-blind full predictions before one evaluation;
   no Phase 1 threshold or curve is a Phase 2 success gate.
