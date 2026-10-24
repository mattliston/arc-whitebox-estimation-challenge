# Phase 2 exact-layer foothold and drift onset

Status: `SOLVED_DIAGNOSTIC__EXACT_L1_NEAR_EXACT_L2_DRIFT_FROM_L3`

## What is exact

Conditioned on the fixed first-layer weight matrix, a Gaussian input remains a
multivariate Gaussian before the first ReLU.  Its componentwise ReLU mean and
its pairwise ReLU second moments therefore have exact Gaussian formulas.  The
next affine map consequently receives an exact first-layer mean and covariance
when those formulas are evaluated without numerical approximation.

Linearity then makes the layer-2 *preactivation* mean and covariance exact.
This does **not** make the layer-2 post-ReLU mean exact: its preactivation is a
weighted sum of correlated rectified-Gaussian variables and is not generally
Gaussian.  Calling the network “two layers shallower” is therefore a useful
engineering intuition, not a mathematical identity.

## What the sealed Phase 2 score shows

The one-use mini100 result for the current CP1792 leader reports:

| Layer | raw MSE |
|---:|---:|
| 1 | `6.881554782955935e-10` |
| 2 | `2.192582806672553e-8` |
| 3 | `2.398551879011119e-7` |
| 16 | `3.722402415224689e-6` |

Thus layers 1 and 2 are already below the `1e-7` objective, while layer 3 is
the first measured crossing.  The Phase 2 correction problem is primarily a
recursive drift problem over layers 3:16, not an initial-condition problem.

## Consequences for learned correction

1. Layer-1 correction must be structurally zero; it is not a learned degree of
   freedom.
2. Layer 2 may be corrected because its post-ReLU law is only approximately
   Gaussian, but it must pass an explicit non-regression check.
3. Training and selection must emphasize stable improvement across layers
   3:16 rather than trading the exact/near-exact foothold for an endpoint win.
4. Teacher attribution must distinguish exact layer-1 identities, the small
   layer-2 non-Gaussian residual, and later recursive state error.

## Frozen evidence

- Joint score file SHA-256:
  `992f8e27d8b44c4ce236051022b4baf43e59b6f610ca57aedb6eb4315b73045c`
- Joint score payload SHA-256:
  `16f93b625d8a779c663c540463deff268c75301bb4aa0e333a2fd7c778758035`
- Candidate:
  `cp_cap_ladder/cp3_cap1792`
- Target rows were opened only by the already-consumed joint scorer.  This
  diagnostic reads the sealed result and does not reopen or persist targets.

Run `python3 verify_exact_layer_foothold.py` from this directory to replay the
receipt checks.
