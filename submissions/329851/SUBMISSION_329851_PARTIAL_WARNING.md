# Submission 329851: adverse partial progress, not a terminal result

September 4, 2026, approximately 22:56 PDT (UTC September 5).
Matt reports the live card shows 23 scored, 5 in flight, and 49 failed.
He expects roughly another 30 minutes before completion and explicitly
clarifies that this is a preliminary read, not an official terminal failure.

The API's status remains `submitted` with the original enqueue message and
null scores; it does not expose those live counts in the bounded poller.
Thus the API status does not prove the job is merely waiting in a queue.

Do not call the runtime repair successful based on three local passes.
At least some remote MLPs can complete, but reliability is not established.
Failure causes are not known yet: do not assume they are all timeouts,
memory violations, or residual-time failures without the terminal evidence.
No further upload while this run remains active.

Current graph training may finish to preserve its in-progress work; new
score experiments are on hold while runtime outliers are investigated.
Response-row experiments reached a local unlearned cost reduction of 5.85%
on rank 35, but no response-row training has started and no matched learned
candidate exists. That path is not a replacement bank.
