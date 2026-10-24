# Fixed Read5 weight-transform reuse port

Parent: `../read5_packet_v1/candidate.py`, SHA-256
`d7723068ae269155e0369e4ba322b447222bbb31295d89394c069695ad0e824b`.
This is a separate candidate; the pending Read5 upload is unchanged.

Port the preserved four-site D5 left-transform cache from
`astra_phase2_reexamination_v1/current_comp_weight_transform_reuse_v1/port.py`.
The old positive component used a different arena layout and did not satisfy its
45-second release gate. Neither its parity nor its timing qualifies this port.

Only the covariance outer product, old-basis push, newborn common push, and star
push may grant reuse of the same physical `W[previous, :].T`. The original shape
dispatcher chooses the kernel. Only unpadded square D5 products can use the cache.
Keep the dispatcher, arena, right transform, multiplication and reconstruction
unchanged. Copy the first D5 left leaf into a separate exact-shape contiguous
array, then omit its six left-transform primitives at each of five levels on
later eligible calls in that layer. The key binds prediction, network weights,
previous-index owner, layer, arena views and left shape. Release all cache numeric
owners after the star push and in a prediction-level finally block. Maximum
additional cache is 68,841,472 bytes. There is no cross-layer/network reuse.

Build append-only definitions and prove the entire Read5 source is an exact
prefix. Also prove the adapted product and each redefined physical function
invert exactly to the corresponding last definition in Read5. The learned
payload and physical estimator do not change.

Before full calls, run truth-free unit cases for 1024x992 left operands and two
right widths (1024 and 384), arena overwrite/interleave, layer and owner
retirement, and original non-D5 fallback. Require byte equality to the ordinary
Read5 product and exact integer billed-work savings, including the creator copy.
No schedule or cache-scope search follows a failure.

Then run full cold/warm ranks 35,100,35 and a separate cold rank100. Require all
16 layers byte-identical to saved Read5 predictions, deterministic repetition,
matching weights, positive billed-work reduction, and retired numeric caches.
If these pass, run the real NumPy-free client on 35,100,35; require the same
outputs and bills, every call below45seconds, and pooled full-context wall time
at most0.99 of the saved Read5 comparison. Source-counted saved primitives are
distinct from measured client requests. A timing failure closes release of this
fixed port, rather than prompting timing-selected variants.

If the current port qualifies, verify the existing14-network panel against
Read5 byte-for-byte before reusing any parent errors to apply the measured new
cold multipliers. This is reused local validation, not an independent accuracy
test. No truth labels are needed for construction or parity checks. No upload
is performed by these scripts.

All numerical work is serial, single-threaded OpenBLAS, nice10 and guarded by
the existing8GB process-tree RSS /256MiB additional-swap supervisor. Preserve
every run and stop a branch on a failed mandatory gate.
