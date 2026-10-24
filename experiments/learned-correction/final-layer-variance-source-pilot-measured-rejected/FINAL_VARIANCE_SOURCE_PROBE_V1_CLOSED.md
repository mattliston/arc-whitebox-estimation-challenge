# Final variance-source pilot — closed

Recorded September 5, 2026, 23:34 PDT; the result file was modified at
23:10 PDT. Phase 2 only. See `final_variance_source_probe_verdict_v1.json`.

All 24 target-free rows passed the unchanged-prefix and decomposition
checks. The fixed matched 20-train / 4-held-out final-head pilot used the
original 1e9-sample mean references, one 40-epoch fit per arm, and frozen P.
The source arm's pooled final raw MSE was 1.0039325113 times the bank and
1.0006956792 times the zero-input control, with only 1/4 bank wins. The
predeclared .90/.92/3-win gates fail. NumPy/MPS head differences were at
most 4.366e-11, so this is not a head-packaging discrepancy.

No 200-row fit, feature/order/layer/epoch/learning-rate/seed sweep,
production implementation, metered candidate, or submission is justified.
This is a local raw-error result, not an adjusted or official score.
