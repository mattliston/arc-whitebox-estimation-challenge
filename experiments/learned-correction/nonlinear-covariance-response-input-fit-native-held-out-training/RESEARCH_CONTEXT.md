# Same-layer response inputs to the learned covariance correction

S31 is the unchanged parent (submission 332517, adjusted 2.1736465056908115e-9). This is a small, prospective training pilot. Qualification or training loss is not a banked score improvement.

The existing edge model has fourteen local physical fields, three pairwise correlation/cumulant terms, and messages transported from the previous layer. The existing node GNN also has row means/norms and a correlation-weighted alpha statistic. Neither explicitly supplies correlations and third cumulants aggregated against the three ReLU response fields used here. The new fields enter the existing nonlinear covariance correction, then propagate through subsequent layers. Both new matrices start at zero, with all existing parameters frozen.

Checkpoint20 already rejected a **linear terminal-mean observer** made from related response messages: TRAIN20 cross-fitting gave +8.6076% raw error, zero winning folds. That was a different consumer, and is not evidence supporting this pilot. The equal-parameter local-field control here tests whether any new gain requires the nonlocal fields.

This is **approximate proposal training**. The loop5js Torch card has known forward differences from S31, and its star basis remains detached. Checkpoint22's full-versus-frozen derivative experiment did not certify a full-model derivative: finite differences disagreed across fixed step sizes and selection branches changed. The native-minus-Torch baseline offset here corrects the initial residual, not the derivative. Only the fixed native CHECK8 assessment can establish a pilot gain.

The source review preceding this pilot also recovered these already-executed directions, which should not be presented as new experiments:

- Uniform intermediate-mean supervision was tried in the covhunter work; intermediate MSE and the final-layer objective were misaligned.
- Low-rank-plus-diagonal covariance replacement failed badly, including the late-only engineer E63 variant.
- Local per-pair covariance prediction in the analyst work had very weak cross-network explanatory power; most closed-loop benefit overlapped the existing rowwise K31 factor.
- Additional leading fourth-cumulant trees provided a small raw gain at material matrix-product cost in the physicist work.
- Free fourth-order coefficients on existing CP columns captured only part of the relevant covariance correction even with tens of thousands of coefficients.
- Thin fourth-order cores and the earlier analytic fourth-plane span had already failed information/cost checks.
- Existing signed and squared weight messages are shared by the node and edge models; their whole cost is too small to promise a large gain merely by thinning them.
- HK233 saved billed work on a few favorable shapes but doubled primitive CPU time, and other shapes increased the bill.

The fixed protocol, model, native builder, calibration, fit manifests, guard receipts, and assessment are the authoritative artifacts in this directory. Do not restart a job based on a progress marker alone. Keep the two fit arms serial, preserve the terminal-only evaluation rule, and do not use CHECK/VALID data to tune a sign, amplitude, epoch, or learning rate.
