# Response-reader qualification, before tiny execution

September 7, 2026 PDT. Addendum to GLOBAL_CP_KERNEL_PREFLIGHT_V1.md,
preserving that prospective plan and all frozen candidate code.

Independent review of candidate_response_rows_current_runtime_v1.py
finds that its intermediate CP off-diagonal read covers
response_indices x live_indices, while _star_compact covers the full
live x live plane. Exact restoration of the underlying tensor's repeated
entries is therefore NOT invariance of future decomposed response reads.
A change that moves a coordinate-star tensor from CP into the explicit
star can change the rows omitted from the CP read. A following linear
push can also expose this decomposition dependence.

The tiny synthetic test is still meaningful for the all-distinct kernel,
signed solve, actual stored-factor restoration and literal primitive bills.
It contains no masked response reader and makes no claim of estimator/head
parity. Its pass cannot close this additional compatibility condition.
Any later score-bearing test must use the exact current-response readers,
all heads and complete rollout, measuring raw and full compute together.
Do not silently substitute full CP rows, switch heads, or call restored
tensor error a bound on the response candidate's score.

No current-state array, model, target, numerical test or candidate has been
executed or modified to obtain this source-level finding.
