# Saved-output localization after the failed response gate

The fixed rank35 response gate failed before scalar calibration or further
native networks. This read-only diagnostic decodes exactly the already-saved
zero,+h,-h,+h/2,-h/2 predictions, plus the repeated half-positive prediction.
No weights, truths, hidden states, new model calls, new amplitudes or fits.

For each of16 layers compute the same central-response stability, signal
and even-part curvature. Null response layers report null ratios. Check
all16 repeat bytes and the first two null response rows. Report the first
layer with response stability>5%, but do not treat output differences as
proof of a selector, QR, clipping, precision or gradient bug. Those internal
states were not captured. This neither rescues the failed radius nor changes
the closed qualification. Its purpose is to locate the observable onset for
future source review, not select another amplitude or candidate.
