# Post-source reconciliation v1

This note preserves `PREREGISTER.md` unchanged and reconciles its pre-source
cost hypothesis with the separately preregistered and frozen CS2 source.
No target, score, dataset row, width-1024 run, package, submission, or graph
edit was used.

The preregistered scientific rule was “enable the scalar pairing carrier iff
all coordinates are valid.”  Its illustrative spelling counted valid entries
and compared with width.  Frozen CS2 SHA-256
`4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779`
implements the exactly equivalent predicate as
`pairing_enabled=fnp.all(valid_high_order)` and applies a scalar fp64 `where`
inside `_reset_stars`.

The distinction changes billing, not the scientific trigger:

```
pre-source estimate: 350*w^2 + 4634*w + 112
measured CS2 fit:     350*w^2 + 4634*w + 98
```

The 14-FLOP correction is explained by fourteen boolean `all` reductions,
each billed as `w-1` rather than `w`, followed by the same fourteen scalar
fp64 `where` operations.  Exact measurements at widths 4, 5, 6, 8, 12, 16,
24, and 32 support the CS2 polynomial; the immutable source result is SHA-256
`c4c8231c17a68f89916fea5fdf87f9dd0fa52be34ad47d03fbe3ab4d0e21bb7f`
and its hold receipt is SHA-256
`c35332c6df5bfb2c7ad7105f200117078d55901956164d7717abdb8f85c2417c`.

At width 1024 the corrected projection is 371,746,914 added FLOPs, total
2,131,652,533,303, utilization 96.9363342530%, and headroom 67,370,722,249.
The literal full-width meter remains authoritative and has not run.

CS2's bounded source probe passed its ten-fixture gates with 2,327
activations, zero parent-finite regressions, three recoveries, and zero stored
carrier invariant violations.  This does not satisfy the held independent
source audit, full-width resource, runtime, moment, or accuracy gates.
