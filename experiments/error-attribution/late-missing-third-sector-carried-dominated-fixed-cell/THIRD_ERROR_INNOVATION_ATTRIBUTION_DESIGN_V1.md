# Is the useful late third residual carried or born here?

Phase2, September5,2026. Diagnostic only, not a new source/fit experiment.

The current source-v2 protocol forces previous means/covariances but not
the full third tensor. Therefore the measured post-third residual is an
ACCUMULATED error. Fresh source corrections failing to align with that
error do not identify how much of it was born at this layer. This
attribution changes the estimand; it does not relax/re-score any failed
source gate or authorize a source implementation.

Fixed35 layer12, same q4 basis and exact two8M seed laws. Let G be the
parent's live-support Gaussian derivative. Acquire just the four PRE-third
slices Tpre(G*U_a) on the same streams. Existing POST-third slices and
independent next-K21 rows are reused, not resampled at another seed/count.
Model pre slices are frozen from the exact current CP+star+trace state.

For each stream define

  Dpost = Tpost_true(U)-Tpost_parent(U),
  Dcarry_a = G*[Tpre_true(G*U_a)-Tpre_parent(G*U_a)]*G,
  Dfresh = Dpost-Dcarry.

Apply the SAME linear q4 projection/coordinate restoration/next-W query
to all three. Certify Dpost=Dcarry+Dfresh after the consumer as well.
Report two-stream reproducible energies, cross-term, cosines and natural
corrections to the original next-plane error. No slope fitting, source-arm
selection, oracle input treated as deployable, or claimed final score.

Interpretation gate only: call this carried-dominated if the fresh
component's cross-stream energy is <=25% of the total and the carried
component has >=.8 cosine with the total, in the fixed weighted consumer.
Otherwise report mixed/fresh without changing the threshold. No rank100,
layer, count or seed expansion follows automatically. A different causal
source test would need a new design and justification after this result.
One low-priority MPS job at a time; no threadpoolctl with Torch.
