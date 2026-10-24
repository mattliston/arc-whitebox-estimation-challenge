# Exact331684 compute frontier: fixed catalog misses standalone3e-9 gate

Both cold predictions reproduce all16 current-parent output bytes and exact
complete bills:413448696909F on35 and416264792412F on100. All five accounting
views independently reconcile every charged operation once, including cold
initialization. The observer calls the original meter once with unchanged
arguments and restores it. Installed libraries and estimator source are
unchanged. Research observer wall times are not production runtime.

The current implementation has295 outermost _mm calls per network,
including two zero-width/zero-bill calls, rather than the older Lazy335.
All observed operands are FP32. Recursive calls are included exactly once
inside their outer bill. Direct products outside _mm remain separately paid.

|Semantic caller,nonoverlapping charge|Rank35 share|Rank100 share|
|---|---:|---:|
|_push_bank_shared_newborn|20.474%|21.235%|
|_shared_old384_read|18.114%|17.907%|
|_slr_reconstruct|13.039%|12.922%|
|_flms_star|12.206%|12.124%|
|_shared_old384_push|10.754%|10.696%|
|_flms_covariance|8.155%|8.099%|
|_slr_inner|8.112%|7.935%|
|_slr_join_factors|2.045%|2.029%|
|gnn_step|1.083%|1.076%|
|_edge_source|0.778%|0.772%|

Native matmul-only shares are79.0853% and
80.2033%. The broader outermost _mm shares
are92.3119% and
92.3708%, including transforms,packing
and other helper operations. Cold reshape shares are
4.5754%/4.3339%.
These categories describe actual cost; they are not independent removable
algorithms. In particular, the shared-plane caller includes young reads and
old outer contraction,while the shared-push caller includes acquisition/QR.
The old reconstruction/inner/rotation group costs about23% of the whole call.

|Fixed catalog|35 optimistic saving|100 optimistic saving|35 with leaf limit|100 with leaf limit|
|---|---:|---:|---:|---:|
|executable_smirnov_and_bank_winograd|13.546294%|14.413668%|12.658075%|13.501748%|
|extended_optimistic_catalog|14.679674%|15.553837%|13.641481%|14.481085%|

The unchanged-accuracy uniform planning threshold for3e-9 is
16.304320666%. Both catalogs fail it on BOTH current development networks,
even before adding omitted implementation work. Extended optimistic totals
are352755774112F/351519644667F; the leaf-limited totals are
357048172304F/355985134441F. The necessary1168900096-byte leaf limit covers
simultaneous leaf A/B/C only,not complete workspace or streaming schedules.

These are finite-family optimistic arithmetic envelopes. They assume
complete-call replacement can remove measured initialization/copy work,
omit new transform buffers,copies,dispatch and finite-precision effects,
and include catalog entries without a qualified native implementation.
No actual saving,runtime,accuracy,official prediction or universal bound
follows. This result closes this catalog as a standalone3e-9 route under
the fixed two-case gate. Smaller improvements,other algorithms and combined
accuracy/cost work are not ruled out. No new kernel is selected automatically.

The integer dynamic program is unchanged from the previous certified search:
at most five levels,minimum leaf dimension8,ceil padding or floor-core/fringe
schedules. Three whole-source inverses prove the limited current-parent,
metadata,call-count and current-yardstick adaptations. There are
4664 independent scalar partition
checks and6 Winograd identities.
The separate verifier replays all2360 emitted
schedules across1180 call records,reconciles
totals and checks catalog-superset monotonicity. No extra search was added.

Earlier Smirnov stream/global-batch runtime failures remain actual failures,
not unexecuted batching leads. The old full-Lazy cost envelope remains
preserved with its own earlier parent and21.39% threshold; neither number
should be substituted for these current measurements.

Two full native calls used829713489321 FLOPs. All four
guards retired,zero added swap; sampled peak RSS
6125158400 bytes,self-reported peak
6161350656 bytes. New artifacts before
closeout total6487051 bytes. No targets,
training,new samples,downloads,GPU,candidate,package or upload.

Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,
100/100 complete,zero failures.4e-9 is banked;3/2/~1e-9 remain unmet.
At unchanged raw error the0.1 multiplier floor is1.949715441540434e-9,
so approximately1e-9 still requires accuracy improvement. The goal is active.
