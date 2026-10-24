# Bounded full-depth readout pilot

2026-09-05, approximately13:12PDT. Phase2 only.

Prerequisite is the full CPU carrier/local derivative certificate, not the
failed frozen-K3 shadow or failed float32 GPU baseline. Forward inference
uses dynamic selections and the frozen bank's scalar-only final head.

Optimize ONLY the existing per-layer mean readouts at layers1..14:
14*(64 weights+1 bias)=910 shared coefficients. Encoder, messages, feature
normalization, gates, all moment physics and the final scalar head remain
frozen. No new inference operation or feature is introduced. The current
mean/covariance/CP/trace physics all participate in the derivative.

Fixed dataset: ranks35..58; new-fit holdouts37,39,42,48; remaining20 train.
These are held out from this pilot, not claimed unseen by every historical
parent fit. Use existing documented1e9 post means, no new labels or samples.
Only final-layer uniform per-neuron MSE is optimized, normalized by the
fixed scale2e-8 for numerical convenience. Intermediate-layer MSE remains
reported so changes are not concealed.

One fixed five-epoch run, seed951212, Adam lr1e-3 cosine to1e-4, gradient
norm cap5. Each layer's combined weight/bias update is constrained to20%
of that layer's original readout L2 norm. This limits the pilot to a local
readout refinement; no hyperparameter, epoch or radius sweep. Save every
epoch immutably, but only the fifth epoch is the prespecified candidate.
Stop on nonfinite values or the30-minute safety deadline. One nice10 CPU
worker, three torch threads; no concurrent MPS or other experiment workers.

After the fixed run, evaluate the final payload using the independent
NumPy actual bank reference on the four holdouts. Expansion requires at
least15% mean raw improvement across these four and improvement on every
one. Otherwise close this bounded readout-refinement pilot. No submission
based on torch training loss, a two-net oracle, or an intermediate epoch.
Any successful branch still needs complete14-net independent4M panels,
candidate/head parity, metered bill and persistent runtime gates before
the preauthorized board submission. Official goal remains<1e-9, bank
milestones<4e-9/<3e-9/<2e-9; current usable bank329926 is4.98223146018e-9.
