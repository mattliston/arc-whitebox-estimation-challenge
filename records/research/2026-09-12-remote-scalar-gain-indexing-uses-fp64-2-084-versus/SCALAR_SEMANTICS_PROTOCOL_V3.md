# Scalar diagnostic v3 — snapshot conversion API only

Incorporate the full unchanged numerical question and guards from the original
protocol and `SCALAR_SEMANTICS_PROTOCOL_V2.md` (SHA256
`e583fae3c64a4b11bcabee21ccab6225320db47a262ed0baf929ea6b6ab03316`).
Frozen worker v2 SHA256 is
`e2645d36e17cc3db5beff1e39a4da2c911df405f65127a594827e0c6f193d32a`.
Both versions and all v2 receipts remain immutable.

The actual v2 local mode passed: 94,780 complete FLOPs and 3,090 per indexed
or sliced expression. The remote mode failed in snapshot serialization before
completing the diagnostic: `RemoteScalar.item` does not exist. The external
stage retired negative in 1.6485 seconds, with 184.8 MB sampled tree peak,
0.22521-second maximum observation interval and no added swap. This is a
harness API failure, not a scientific falsification or a positive remote bill.

The actual pinned client `_remote_array.py` SHA256 is
`8c74bfdcd08502d1b53daac4aa0b8c24809aaa1408c17e6178deb83a9876fd54`.
`RemoteScalar.__float__` at lines 397–398 is exactly
`return float(self._value)`. `item()` at line 1165 belongs to `RemoteArray`,
not `RemoteScalar`. V3 authenticates and AST-checks the correct class/method.
Only the by-value scalar snapshot changes from `value.item()` to
`float(value)` before the existing `struct.pack` at the already reported dtype.
It neither dispatches FNP arithmetic nor replaces or narrows the original
object passed into the next tested expression. Actual budget-before/after
checks around snapshots remain unchanged.

The full v3 body reverses byte-for-byte to the frozen v2 derived body after
five unique changes: that snapshot call; schema suffix v3; fresh mode output
directories; v3 protocol filename; and additional v2 source/protocol pins.
No tested arithmetic, constant, dtype, vector, gain, ordering, hypothesis,
complete-cost ownership, resource guard or comparison changes.

CLI is `scalar_semantics_probe_v3.py --run --mode local|remote
--source-sha256 V3_WORKER_SHA --protocol-sha256 V3_PROTOCOL_SHA`, with the
unchanged remote `--server-url` requirement. Optional `--out` must match
exclusive `scalar_semantics_v3/local` or `/remote`. Schema is
`current_comp_shared_old384_scalar_semantics_v3`. Each mode produces fresh
linked result/completion/finalization seals; no old local result is substituted.
The parent owns the versioned watcher, same whole60s/tree2GB/noSwap/actual1s
sampling and all owned IPC process-group cleanup.

Source/API inspection, stdlib derivation/compile and `gain_source()` metadata
preflight are permitted without numerical imports or stages. No new numerical
execution is authorized by writing these sources. Full estimator fidelity,
candidate change and release remain unestablished regardless of this tiny test.
