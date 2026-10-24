# Target-free allocator ownership diagnostic

This is a separately versioned implementation experiment following the closed
current_full_lower_p_proposal_v1 resource failure. That failure, including its
8,583,757,824-byte sampled RSS, remains a failure. No training, score labels,
candidate, changed objective, precision, rank, arithmetic or optimizer update is
authorized by this protocol. The official bank and pending release are unchanged.

The sole intervention is Python garbage collection and the installed Apple
malloc_zone_pressure_relief(NULL, 0) API at model step boundaries and between
calls. The installed malloc.h documents this as a request to release unused
mappings in all malloc zones; it does not free live allocations. Its return value
is distinct from measured RSS reduction. All-zone malloc statistics, current
process RSS, process high water, and weakly tracked state-storage lower bounds
will distinguish live ownership from reserved mappings. Weakly tracked boundary
storage is not a complete count of C++ autograd, library, or process memory.

The original model_v2.py is imported unchanged, with its original common.py.
No translated function, tensor operation, layer state, checkpoint schedule or
gradient formula is changed. This remains an approximate ordinary-product
proposal engine, not a native derivative. All prior accuracy failures retain
their original scope and verdicts.

Before wide work, compile a tiny C fixture against the pinned installed header.
Check sizeof and every field offset of malloc_statistics_t against ctypes, call
the documented APIs, and verify live NumPy/Torch array bytes and deterministic
small forward/gradient bytes survive pressure relief. Test the stricter guard
on a small owned process group that deliberately exceeds a small test limit.
All sources, input metadata, controls, header, compiler and C fixture binary are
sealed before numerical tests.

The new guard triggers at 7.5 billion bytes of tree RSS, samples every 0.05s
(recording actual intervals), stops on >256MiB system swap growth, and requires
zero observed added swap for qualification. It retires the entire owned group.
Only one numerical worker group runs at a time, nice 10, all six BLAS thread
variables and Torch intra/inter-op threads set to one, CPU only. No allocations
near the cap are used in guard tests. Available disk must remain >=470GB and
new study artifacts <=1GB. Unit stages <=180s; full qualification <=1800s.

The wide sequence is fixed: no-grad rank35, checkpointed rank35, checkpointed
rank100, checkpointed repeat rank35. Use the original synthetic cotangent seed
940921 and the same zero-learning-rate Adam initialization and call order. After
the plain call, require a process peak <=6.5GB and retired current RSS <=2GB before
attempting backward. These are conservative launch conditions, not a proof of
backward resource safety. The independent process guard remains active.

All rank35 outputs must equal the completed old plain35 output byte for byte;
all recorded rank35 domains, selectors and shared-state geometry must equal its
record. Repeat35 gradient hashes must match. All eleven parameter gradients
must exist and be finite; inactive layer-zero coordinates must remain zero.
All parameter and fixed-payload bytes must remain unchanged through zero-rate
Adam state allocation. Cards must retire between calls. Only a completely
finished sequence, all checks, and a clean guard receipt qualify this resource
implementation. Any guard stop, mismatch or incomplete backward closes this
fixed attempt without a fit. A future recomputation schedule requires its own
version and qualification; no automatic rescue is part of this protocol.

Even a passing resource result is not an accuracy result or native-gradient
certificate. The old fixed one-pass training proposal would need a separate
data/fit seal and actual native score/cost endpoint before release eligibility.
