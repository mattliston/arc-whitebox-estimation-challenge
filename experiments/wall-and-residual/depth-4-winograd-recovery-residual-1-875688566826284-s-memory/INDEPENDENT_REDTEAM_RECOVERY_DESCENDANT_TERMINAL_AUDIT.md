# Independent red-team terminal audit: P2R0C1D5R1

Date: 2026-08-26

Verdict: **CLOSED_THIS_IMPLEMENTATION_FORM. FAIL_TERMINAL_NO_RETRY.**

The exact one-shot target-free descendant stopped correctly after candidate
ordinal 0 failed the frozen residual-wall and RSS gates. It wrote a valid
terminal result/receipt, exited 1 with the expected no-retry `RuntimeError`, and
did not execute ordinals 1 or 2. No reference estimator call, target access,
score, production action, or submission occurred.

## Sealed terminal evidence

- transcript `RECOVERY_DESCENDANT_EXECUTION_TRANSCRIPT.txt`, SHA-256
  `afa66d0a953b4d3f0d703bbbb62321dfd3b10df0aec7b716ec6847e3ddaf68b3`
- terminal receipt `RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json`, SHA-256
  `8ee4d2be92f780b4f547e2129e59241cb19936a9767df1f04b3e9b6ba9b281c0`
- runner result `RECOVERY_DESCENDANT_RESULT.json`, SHA-256
  `a8221d116c03ed442cb6343a54b429b06e5a620f905c056fa99abe6497bc9549`
- ordinal-0 output `recovery_candidate_rank35_first.npy`, SHA-256
  `4a38b23d7fb995a005566336c0f7d3c5adac7cf46a5d02d6cba4b7c598b15290`
- terminal verifier `verify_descendant_terminal_execution.py`, SHA-256
  `cb9fbb896d004378f658ccb012bc94dfd59073bc81a3a50eb36e2d913104dd82`
- transcript extractor `recover_descendant_terminal_execution.py`, SHA-256
  `5f53e72b26c330717e8da7af830a0bc3a8299b8ffff8797b61a6ea08e622c2c3`

The terminal verifier passes and the preserved session-event hashes establish
the single launch, one printed trial, terminal status line, traceback, and exit
code 1.

## Exact ordinal-0 result

| Measure | Result | Gate |
|---|---:|---|
| Literal candidate FLOPs | 1,320,876,655,991 | pass, exact |
| Literal operation sum | 1,320,876,655,991 | pass, exact |
| Measured reference FLOPs | 1,692,484,904,311 | bound |
| Measured saving | 371,608,248,320 (21.9563700316%) | pass, exact |
| Outer wall | 24.356781082693487 s | pass, <120 s |
| Meter wall | 24.35678375000134 s | pass, <120 s |
| Residual wall | 1.875688566826284 s | **fail**, not <0.4 s / not <=0.385 s |
| Peak process RSS | 11,953,569,792 B (11.1326293945 GiB) | **fail**, not <8 GiB |
| Final-layer RMS drift | 4.109649235540117e-8 | informational/per-call accuracy pass |
| Final-layer maximum absolute drift | 2.384185791015625e-7 | pass, <=8.574549521806375e-6 |

The output is finite float32 with shape `(16, 1024)` and independently
recomputes the recorded drift and array hash. The arithmetic/accounting result
is real: exact call-0 saving is 371,608,248,320 FLOPs and numerical drift is
small. But deployability fails decisively:

- RSS is 3,363,635,200 bytes (3.1326293945 GiB) over the strict cap and
  3,916,464,128 bytes (3.6474914551 GiB) above the corrected analytical peak
  projection of 8,037,105,664 bytes.
- Residual wall is 1.475688566826284 seconds above the 0.4-second gate, or
  4.6892214171 times the limit.

The prior static memory model is therefore not a valid launch authority for a
nearby successor without a materially stronger accounting of live NumPy
temporaries and process RSS. Literal RSS remains decisive.

## Scope of what was and was not learned

Only rank-35 ordinal 0 executed. Therefore the rank-279 bill/drift and repeated
rank-35 candidate parity were not measured, and neither pooled RMS definition
nor aggregate maximum/repeat gate was reached. The ordinal-0 repeat flag is
vacuously true by runner construction and is not a repeatability result. Do not
describe this as a full `[35,279,35]` accuracy pass or as a raw-MSE result; no
targets were opened.

This closes the implemented paired/sequential depth-4 shaped-scratch schedule,
not the mathematical possibility of all Strassen-like CP3 reductions. Any
successor must be materially different and preregister, before execution, a
credible route to reducing the observed peak by more than 3,363,635,200 bytes
and residual wall by more than 1.475688566826284 seconds. Merely rerunning,
retuning thresholds, switching fixtures, or applying the old static margin is
not authorized. A successor must also retain exact FlopScope accounting and all
accuracy, repeat, target-free, wall, RSS, and no-retry gates.

No further D5R1 execution is authorized.
