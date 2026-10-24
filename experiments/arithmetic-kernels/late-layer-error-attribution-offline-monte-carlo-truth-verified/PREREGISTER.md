# P2R0G19A: late-layer mean-error attribution vs offline MC truth

## Question
Which mechanism owns the Q24 host's late-layer (l=10..15) per-layer mean error:
(A) order-2 marginal emission truncation at TRUE pre-state (fixable only by a
higher-order/width-finite closure), (B) pre-state error echo (fixable by better
state propagation), or (C) host emission-convention difference vs plain order-2?

## Frozen scope
- Rows: development ranks 35 and 100 (feature-store weights only; NO targets,
  NO observer_selection; truth = offline MC from weights, seeded).
- Truth: P = 8,000,000 seeded standard-normal inputs per row (PCG64DXSM,
  seed = mlp_seed ^ 0x47414D454F4E5F31), float32 forward, float64 accumulators:
  per-layer post-mean, pre-mean, pre-var, pre-k3, pre-k4 (+SE estimates).
- Host: workspace_v4 estimator host path (predict_with_native_tape), with a
  read-only tap of the per-layer univariate consumption state (m,v,k3,k4).
- Exact telescoping decomposition per layer l:
  e_l = mMC_post - m_host_post
      = [mMC_post - Edg2(true pre)]      (term A)
      + [Edg2(true pre) - Edg2(host pre)] (term B)
      + [Edg2(host pre) - m_host_post]    (term C)
  with Edg2(m,v,k3,k4) = Gaussian ReLU mean + (k3/6s^3)c1[3] + (k4/24s^4)c1[4]
  + (k3^2/72s^6)c1[6], c1 the Stein columns.

## Metrics and gates
- Per-layer noise-corrected error energy ||e_l||^2 - 1024*sigma_MC^2/P and the
  three term energies + pairwise cross-energies, l = 10..15.
- Attribution gate: the dominant term (or named pair) must cover >= 70% of the
  summed noise-corrected late-layer error energy on BOTH rows to direct G19B.
- Authority gate: MC noise energy <= 65% of raw error energy per layer
  (else escalate P; do not interpret).

## Outcome split
- A dominant -> G19B pursues higher-order/width-finite emission closure.
- B dominant -> G19B pursues state-propagation accuracy (covariance/planes).
- C material -> host convention audit first (cheap).
- No term >= 70% (mixed) -> record split fractions; G19B attacks the largest.

## What this does not prove
No MSE improvement; no deployment feasibility; development rows only.
