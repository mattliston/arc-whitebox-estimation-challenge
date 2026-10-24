# Local server configuration diagnosis, before retry

September6,2026,19:37PDT. The first local persistent-series invocation
failed before layer0 while allocating the bank's unchanged scratch arena:
1168900096bytes exceeded the standalone server's generic100MiB default.
No MLP completed. All v1 JSON/resource/log artifacts are preserved.
Owned server/client exited; no estimator or backend source was changed.

The installed server's `_request_handler.py` reads
FLOPSCOPE_MAX_ARRAY_BYTES with default100*1024*1024. The already-frozen
REMOTE_DTYPE_BUG_2026_09_04.md and STATUS_AFTER_329884_2026_09_05.md
record exactly this standalone-harness mismatch and the prior bank tests'
FLOPSCOPE_MAX_ARRAY_BYTES=4294967296 configuration.

Read-only primary-source verification found the organizer's
[memory-cap announcement](https://discourse.aicrowd.com/t/memory-usage-limit/18039):
the grader's per-array guard was raised to4GiB. This is platform resource
configuration evidence, not Phase1 estimator work. The current
[Phase2 announcement](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197)
still specifies8GB process memory,120s total wall and0.4s residual, all of
which remain in force. No newer per-array reduction was found. The current
bank's passing grade also used this unchanged1.169GB scratch allocation.

The v2-tag retry will use the same reviewed run_remote_series_v1.py and
check_remote_series_v1.py, same candidate/source5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172,
same five contexts and one BLAS thread, explicitly restoring only the
previous4GiB per-array test-server setting via its documented environment
variable. No accounting, transport, estimator, model, threshold, score,
memory-release threshold, or wall limit is relaxed. Keep the stricter
combined sampled RSS<8e9, local wall<60s and residual<0.4s release gates.
This scoped local diagnostic setting is not bundled or sent to the grader.

Successful completion, actual bill and runtime safety are still unproven
at this point. This setup correction does not constitute another fitted
arm or tuning under the response-row exception.

Bookkeeping correction from filesystem inspection: this note was first
written at19:36:10PDT, not the19:37 minute stated above. The v2 launch
followed it; the tool clock read02:36:23UTC after launch. No output or
result existed when the retry configuration was recorded.
