# Explicit follow-up after a failed primitive CPU gate

V1's fixed primitive gate FAILED: aggregate CPU 2.3351549x S31, although
aggregate bill was 0.9419900x and all error gates passed. The public einsum
backend follow-up also failed (CPU 1.0390609x matmul). Neither is reclassified
as a pass; neither justifies submission.

The primitive CPU ratio is an engineering screening heuristic, not the
platform's whole-prediction time limit. It is unweighted across four shapes
and cannot determine how much of a complete estimator's work changes.
To resolve that limitation, permit one fixed full-program TRAIN35/100
diagnostic of the unchanged v1 arithmetic and routing. This explicitly
supersedes the earlier requirement that the primitive speed gate pass
before any full-network run. It is a deliberate protocol amendment before
any new endpoint labels or scores, not evidence that the old gate passed.

No depth, shape, coefficient, cutoff, padding, capacity or route is tuned.
The standalone builder appends v1 verbatim to S31 and registers the three
private caches in the parent's existing reset lifecycle. That addition
only prevents obsolete scratch views from surviving a setup reset.

Use original complete-network score gates: both adjusted wins, pooled
improvement at least 1.5%, pooled raw degradation below 0.5%. Also record
complete CPU/wall time, routing counts and memory. These two reused TRAIN
networks are a diagnostic only, never validation or a platform runtime
qualification. Any next expansion must use the frozen source. No upload
without meaningful improvement, validation and actual-worker checks.
