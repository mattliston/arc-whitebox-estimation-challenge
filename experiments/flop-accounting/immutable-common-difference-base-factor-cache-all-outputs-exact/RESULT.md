# Retained exact UVP base B cache

Measured mean charged compute decreased **0.1158296%** versus banked S32 on the fixed35/100 pair. All16 prediction layers were byte-identical in native evaluation and after promoting actual RPC float32 outputs to float64. Complete billed FLOPs matched between native, worker and boundary paths.

The cache reuses only three transforms belonging to the same immutable join10 F generation, including its unchanged lazy base at join13. Explicit native slice/padding ancestry provides offsets. Each network recorded13 hits and3 counted copies, with336,946,736 bytes of cache storage, then released all cache ownership. Both networks saved319,701,476 FLOPs and296 requests.

Guarded native2, actual worker2 and boundary2 completed and retired with zero added swap. Peak group RSS was5,996,740,608 bytes; peak handles48,637. Setups were3.038/3.124s with624 bindings and600 fused entries. Boundary residual deltas were-0.0004/-0.0011s; these timings are descriptive. Mean boundary CPU ratio was0.99015 versus the saved matched S32 process.

Candidate SHA256: `0308a99fd89823cfd67e19c3ad00c946d4211b6c69d956519267e142aa205fd5`. Root source review and independent cost review passed; exact inverse of six metadata edits plus removal of the appended cache restores S32. Source and receipts are pinned by `retained_component.json`.

This is a retained component, **not a new board submission**. No fullworker20 or broad score validation has been performed. The >=1% aggregate submission policy remains in force. The unbanked D3 setup component is separate and has not been composed with this source.

Historical correction: the old factor-cache idea was skipped on resource forecasts or a former1.5% implementation threshold; it had not failed an implemented cache test. Current source geometry made a bounded version worthwhile.
