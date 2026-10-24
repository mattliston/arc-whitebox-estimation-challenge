# Fixed warm-start suffix training pilot

September 5, 2026. No fit has been inspected. Full-depth approximate
gradients are rejected; this experiment freezes the entire real prefix
through layer 8. The simultaneous layers 9..14 response check passed on
35/100: relative error .036597/.034360, full half-step .009762/.020681,
shadow half-step .000784/.001069. Separate adjoint checks also passed.

Use public ranks 35..58, with held-out 37/39/42/48, i.e. 20 train / 4 valid.
Generate target-free baseline tapes using the unchanged current full-pair
carrier and payload. Previously generated rank 35 tape is reused, not
overwritten. No covariance labels or sealed targets are opened. Existing
1e9 post-mean labels are loaded only by the training/scoring stage.

Warm-start the existing GNN parameters for layers 9..15; centers, scales,
RMS normalizers and the whole prefix remain frozen. Shared message/encoder
weights may change only in the suffix's separate payload. Train through
the certified mean/covariance shadow, holding K3/K4 responses and masks
fixed. Five epochs, Adam LR 5e-5 cosine to 5e-6, zero weight decay, gradient
norm cap 5, seed 710298. Loss is the clipped reported mean MSE over layers
9..15, final layer weight 4 and other layers weight 1, normalized by TRAIN
parent final MSE. No best-epoch selection, LR sweep, or later-epoch extension.

Only the fixed epoch-5 payload is evaluated in the ACTUAL full carrier
on the four held-out nets, with dynamic K3/K4/masks and the original prefix.
Require >=8% raw final MSE reduction on EACH held-out net versus its exact
parent trajectory, scored against the same stored 1e9 means. Otherwise
close this fixed warm suffix trial without 200-net training, port, final
scalar-head refit, 14-panel evaluation, or submission. Passing would only
justify the next matched full-data/resource stage, not a bank or milestone.

At most three nice-10 row workers; training starts after generation and is
the sole MPS job. Stop the pilot if training becomes nonfinite or exceeds
15 minutes. Preserve all partial artifacts and receipts.
