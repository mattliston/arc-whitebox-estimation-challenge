# What the resource failure does and does not establish

The original layer-checkpointed CPU implementation failed the8GB tree-RSS
qualification. The guard's largest observed sample was8,583,757,824B. It
terminated and retired the worker during the first backward pass, with
zero added swap. This is a measured resource failure, not an accuracy result,
a completed gradient, or an assertion that the actual peak was exactly
that sampled value. The original fixed proposal does not proceed to fitting.

The plain rank35 forward completed in6.0991s, process peak6,793,330,688B.
The checkpointed rank35 forward reached its logged completion in5.7168s,
reporting process peak7,262,781,440B, before the backward guard stop. Only
the plain output and its complete diagnostics were finalized. Its final
RMS discrepancy from the current native bank is4.08708296561e-6; no score
label was read. No backward/repeat qualification or numerical derivative
claim follows from that one completed forward.

The saved boundary geometry permits one exact storage calculation without
decoding any additional arrays. Across the16 step inputs, the logical full
signed-factor p tensors contain869,007,360 bytes. At entries6..15, the old
prefix up to stops[layer-6] is never read by the next shared push: that push
reads only the joining and younger suffix, while older factors are generated
from Q,F and core. Summing those unread prefix slices gives428,081,152 bytes.
This is logical tensor payload, not a measured reduction in RSS, and it is
smaller than even the observed584MB overshoot. Removing those slices alone
has no resource qualification or sufficient measured margin.

The no-grad forward's6.79GB high-water mark also does not identify how much
memory remains live. Saved autograd states, temporary arrays, allocator
retention, fragmentation and library workspaces cannot be separated from
ru_maxrss. The next useful resource experiment should measure their ownership
before modifying physical precision, rank, loss, data or inference behavior.

The installed Apple SDK header declares malloc_zone_pressure_relief at
malloc/malloc.h lines195..200. Its documented NULL-zone, zero-goal operation
asks all malloc zones to release unused mappings and returns the number of
bytes released. This is a possible implementation-only diagnostic; it has
NOT been called here, and no RSS reduction or successful backward is claimed.
Any new such experiment must be separately versioned and sealed, retain
the failed original, prove unchanged output/parameter bytes, and use a more
conservative7.5GB trigger with faster observation. Recursive recomputation
is another implementation possibility, not an executed memory solution.

The full goal remains approximately1e-9. A future resource-qualified version
would still need its fixed fit and actual native score/cost endpoint. This
note licenses no reuse of a failed qualification flag or automatic training.
