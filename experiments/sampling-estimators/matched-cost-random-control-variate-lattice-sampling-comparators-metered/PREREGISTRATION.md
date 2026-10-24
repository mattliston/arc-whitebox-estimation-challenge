# P2R0G equal-budget sampling comparator

Status: frozen target-free resource discriminator; Phase 2 accuracy unopened.

## Fixed comparison point

Every lane is compared with the existing CP1792 receipt, not with the full
`2^41` hard cap and not merely with the `0.1` score-floor ceiling.

- CP1792 source SHA256:
  `035e3bd2fd08d3358f5424fc67f10281e5559e65fa9c6d6dce97791108b176dc`
- CP1792 receipt SHA256:
  `71c4eedf2b04f3430bbca3ebf20882b0dc08daadf8895be14112f85fca240115`
- CP1792 flopscope-0.12 bill: `214,122,341,378` FLOPs.
- CP1792 residual wall time: `0.13626903528347611` seconds.
- CP1792 sampled descendant peak RSS: `2,371,272,704` bytes.

For a sampling lane, equal budget means the largest frozen structural sample
schedule whose measured official bill does not exceed the CP1792 bill.  A gap
is admissible only when the next indivisible structural unit would cross the
comparison bill.  No dummy work may be added to manufacture equality.

## Frozen lanes

1. `EstimatorIID`: 6,372 float32 IID Gaussian inputs and raw layerwise sample
   means.
2. `EstimatorRQMC`: independent continuous random shifts of nested base-two
   rank-1 lattice blocks `(2048, 2048, 2048, 128, 32, 16, 8, 4, 2)`.  Every multiplier
   is odd, so each coordinate of every `2^m` block is a permutation of its
   one-dimensional `2^m` grid before the random shift.  Normal conversion uses
   the counted flopscope `norm.ppf` implementation.
3. `EstimatorAffineCV`: 3,183 IID inputs propagated alongside an affine path.
   If the deterministic anchor at layer `l-1` is `a`, the affine update is
   `s_l = a_l + g_l * (s_{l-1} W_l - a W_l)`.  Therefore
   `E[s_l] = a_l` inductively from `E[s_0] = 0`, independent of whether the
   diagonal-Gaussian anchor describes the nonlinear network well.  The reported
   estimator `a_l + mean(h_l-s_l)` is an exact-expectation control-variate
   spelling with no fitted coefficient.

All streams are domain-separated deterministic functions of the official
grader-supplied `mlp.seed`.  The source contains no target identity, target
value, score, learned weight, Phase-1 checkpoint, or target-derived choice.

## Frozen execution order and gates

1. Static/source verification only.
2. Official dataset-free `whest run --runner subprocess` receipts under
   whestbench 0.16.0 and flopscope 0.12.0, retaining raw stdout, stderr, and
   10 ms process-tree RSS traces.
3. If and only if all three bills are at or below CP1792, run one target-blind
   source-weight discriminator on a pinned mini row.  It may record predictions,
   hashes, cross-estimator disagreement, nested/repeated-scramble disagreement,
   FLOPs, wall decomposition, and RSS.  It must not decode any target column or
   compute any target error.
4. Accuracy remains `HOLD` until a separately authorized sealed scorer compares
   the frozen predictions with protected targets.  No lane may be selected,
   fitted, resized, or retried from those target outcomes.

Resource PASS requires `(16,1024)` finite float32 output, official bill no
greater than CP1792, residual wall time below 0.4 seconds, sampled descendant
RSS below 8 GiB, and total wall time below 120 seconds.  Scientific promotion
requires lower sealed layer-16 and all-layer raw MSE than CP1792 at the same
official-bill rule; no public target value is authorized by this document.
