# CHECK4 v1 terminal validation failure, preserved

2026-09-12, Phase2. Candidate4d98fef2 is unchanged. All five native calls
37/39/42/48/37 saved their outputs, complete own bills, diagnostics and actual
context-retirement markers. No CHECK4 raw score or truth read occurred.

The worker then failed `probe37_check`: final warm37 operation_count59823
versus59803 in the original immediate-repeat probe. Complete warm FLOPs are
identical404454713141, as are output NPY SHA9921bf5d and payload SHAa43c0a57.
Cold37 output and FLOPs406399683439 also reproduce exactly. The all16/47
domain/selector packets, startup packet and all24 warm D6 dispatch records,
including their own operation counts, match the original probe. This does
not by itself identify the extra20 operations.

Source diagnosis: the original packed-frame cache is a shared FIFO128.
Each cache-frame build has four public unstack calls per recursion level;
official flopscope records each unstack with zero FLOPs. One additional
depth5 frame rebuild explains +20 operations and0F. The extra three networks
change this cache's history. The precise evicted key was not captured in
v1, so this remains a source-supported inference, not an observed op trace.

Worker full interval307.1957478756s; watcher308.6154891253s, five observed
retirements, complete-tree peak6343049216B, max completed observation interval
.4389762497s and no added swap. All owned and observational groups retired
in the watch's final cleanup. Final worker and watch chains remain FAILED;
resource evidence is not relabeled a positive numerical qualification.

Preserved result/completion/finalization hashes:

- worker:7f43c2d8b4bacf267b75b044ee70532babf8979c1b3c1813cd2febe92e3ab385;
  63e1977e88768389d7324cdc590f82db1d07c4409fed0edab4b92774ff3bd07c;
  c1f02631200437a8fc69a101d7726790c4070c0a6239bfa4c02d3173c5c4f67b.
- watch:c65408aeeaee72263e06824648e95060951a96fa888d08d76011f0b7911717df;
  61371b005c18ee172c7396740f013c2aeb41e539bf03e4e6d25128a6b58d3e65;
  8644aa34984c6c7acc4da2e1d0871a7832e779770939c5834ac91907a545fc00.

Decision before any CHECK4 target read: make an additive fresh v2 with the
same candidate, paid loop, five networks, complete FLOP/output/domain/dispatch
checks and resource limits. Record history-dependent operation counts,
and passively summarize the closed context's operation ledger. No tuning,
fitting, new estimator arithmetic or scientific gate change. The old failed
receipts are preserved, not repaired. A fresh positive watch is required
before the unchanged .97/3-of-4/1.10 adjusted-score test can execute.
