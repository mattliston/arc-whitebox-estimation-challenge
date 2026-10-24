# Exact optimizer-state handoff passed

All three numerical workers and the final verifier completed successfully.
The study executed 16 backward graphs and four optimizer steps in total:
eight graphs/two steps in the uninterrupted reference, then two fresh
four-graph/one-step workers. All process groups retired.

Every reference/split prediction, accumulated gradient, clipped gradient,
parameter hash, variance-domain record, selector, state shape and checkpoint
visit matched exactly. Both optimizer boundaries matched all parameter,
Adam step/moment, group-setting and CPU RNG bytes. Every NPZ state passed
an exact round trip without changing live values. The resumed worker used
the first fresh worker's complete boundary state.

The reference also reproduced the failed single-process fit's first eight
graphs and first two updates exactly. No failed-fit partial payload was
loaded. No validation target arrays, native evaluation or candidate export
were involved.

Maximum observed group RSS was 6,009,126,912 bytes. Maximum post-graph
retirement RSS was 1,406,697,472 bytes. Added system swap was zero. The
unchanged 7.5 GB peak and 2 GB retirement limits passed. Total controller
wall time was 507.031 seconds; session 24419 exited with status zero.

This establishes the tested eight-graph state handoff only. It neither
identifies the earlier memory-growth cause nor proves full-fit resource
success, native gradients or accuracy. A separately sealed full fit must
start from the original zero branch, preserve all 40 fixed updates and
verify the entire 134-graph/33-update completed old prefix.

The authoritative receipts are `qualification.json`, `result.json`, the
four worker guards and their pinned graph/update/checkpoint artifacts.
