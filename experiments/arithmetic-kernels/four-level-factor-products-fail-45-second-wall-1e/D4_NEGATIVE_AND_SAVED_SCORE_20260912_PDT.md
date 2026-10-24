# D4 component result: not qualified, preserved for diagnosis

Actual watch_d4_only.py execution completed both D4 calls but ended negative
at the all16 vector tolerance. No source, head, threshold or receipt is changed
after this result. The prior baseline-only failure also remains negative.

| Same canonical37, actual NumPy-free RPC | Cold | Repeat |
|---|---:|---:|
|D4 wall seconds|48.457078958|49.614106792|
|D4 backend seconds|36.106953350|36.622258981|
|D4 FNP overhead seconds|12.111948605|12.730520331|
|D4 residual seconds|0.238177003|0.261327480|
|D4 complete served F|416,379,396,797|414,995,482,499|

Own initialization1,383,914,298F matched the prospective price. Both all16
output byte hashes:e670d99ad79982f68bcf368abed0beb39353c5df05eea3335d2c9a8beb06e18e.
The preceding current-c2 cold call was62.916841667s and406,399,683,439F.
D4 therefore cut cold wall about23.0%, but missed both-calls<=45s gate.

The all16 maximum difference versus current c2 was3.0487775802612305e-5,
above1e-5. L0/L1 exact; maximum difference grows from1.19e-7 atL2 to1.97e-5
atL12 and3.05e-5 atL15. Final prediction-difference MSE5.622208958485086e-11.
Thus D4 cannot inherit an accuracy-preserving or all47/native/RPC qualification.
The stage failed before setting its information/ratio flags; none are inferred
as passing from its two saved completed calls.

Actual stage101.04461s, full owned peak5,411,536,896B, maximum sample0.550200s,
zero added swap; both client contexts closed and all owned/observational groups
retired. Negative result/completion/finalization hashes:

- 8e82d1215dbe161122c08085f9985409e35e4c40dc0b194ffe8dbf2552c5fc49
- 2e28c389bde1af2c702adee69181d53bfea1e089a70c9843ce35f3c29f16bd2a
- c6aed9bc3b179e126f79074c57189a92758f1a9b333067d829d3ac7244cd9da9

## Saved-output diagnostic only

After retirement, a read-only stdlib calculation authenticated all three
output payloads and actual bills before accessing only the already-pinned
ordinary1B rank37 truth. It widened the original FP32 served values and truth
to Python doubles and used fsum of squared differences. Current c2 reproduced
its original raw value exactly. No estimator import/run, refit, or promotion.
Runtime0.005759s, self peak22,085,632B, ten-second alarm, no files written.

| Rank37 ordinary1B diagnostic | Raw final MSE | Complete cold F | Adjusted |
|---|---:|---:|---:|
|current c2|2.1882384000672836e-8|406,399,683,439|4.044065431464368e-9|
|D4|2.1742137918149484e-8|416,379,396,797|4.1168178865683616e-9|

Raw ratio0.9935909139278865; adjusted ratio1.0179899302661999. This is one
saved-output diagnostic, not a new panel bank, positive qualification, or
official result. D4 alone is not a score improvement. Its measured speedup
can be preserved as information for a differently validated combined estimator,
but none of the failed original component gates are retroactively relaxed.
The prepared pair256+D4 composer remains source-only and conditional; it has
not acquired an admissible positive D4 component.
