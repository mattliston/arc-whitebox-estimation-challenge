# Setup budget

The 5 s setup cap and prebinding.

| Date | Experiment | Programs | Outcome |
|---|---|---:|---|
| 2026-09-23 | [Setup-time fixed-frame binding saves too little average work for its setup-time risk](setup-time-fixed-frame-binding-saves-too-little-average/README.md) | 8 | rejected |
| 2026-09-25 | [Repeated-shape setup prebinding saves 1.7% cold work; all five platform workers complete setup in about 2 seconds](repeated-shape-setup-prebinding-saves-1-7-cold-work/README.md) | 4 | adopted |
| 2026-09-25 | [Increasing tree-cache caps reduces rebinding work while preserving predictions](increasing-tree-cache-caps-reduces-rebinding-work-while-preserving/README.md) | 4 | adopted |
| 2026-09-25 | [Setup guard can skip every contraction leaf buffer; binding those buffers first is staged](setup-guard-can-skip-every-contraction-leaf-buffer-binding/README.md) | 5 | adopted |
| 2026-09-25 | [Value-ordered prebinding and cache cap 480 reduce repeat binds while respecting setup limits](value-ordered-prebinding-cache-cap-480-reduce-repeat-binds/README.md) | 6 | adopted |
| 2026-09-25 | [Contraction leaf buffers bound before optional frame fusion preserve complete setup coverage](contraction-leaf-buffers-bound-before-optional-frame-fusion-preserve/README.md) | 4 | adopted |
| 2026-09-26 | [Positive-tail trace and 600 setup keys: held-out adjusted score improves 1.73538%](positive-tail-trace-600-setup-keys-held-out-adjusted/README.md) | 1 | adopted |
| 2026-09-28 | [Nine depth-three setup bindings: requests fall by 996 cold and 870 warm](nine-depth-three-setup-bindings-requests-fall-996-cold/README.md) | 13 | adopted |
| 2026-09-28 | [Twenty-seven additional setup bindings: cold bill falls 1,493,091,978 FLOPs](twenty-seven-additional-setup-bindings-cold-bill-falls-1/README.md) | 18 | adopted |
| 2026-09-30 | [All setup keys fused eagerly: degraded setup drops bindings and increases billed compute](all-setup-keys-fused-eagerly-degraded-setup-drops-bindings/README.md) | 10 | rejected |
| 2026-09-30 | [First 128 setup keys fused eagerly: degraded setup retains all bindings and exact FLOPs](first-128-setup-keys-fused-eagerly-degraded-setup-retains/README.md) | 182 | adopted |
| 2026-09-30 | [Nine essential setup fusions: forced-skip requests fall by 870 with limited residual improvement](nine-essential-setup-fusions-forced-skip-requests-fall-870/README.md) | 9 | inconclusive |
| 2026-10-01 | [Refreshed setup-key census raises fused-tree usage from 56.7% to approximately 75%](refreshed-setup-key-census-raises-fused-tree-usage-56/README.md) | 2 | adopted |
| 2026-10-02 | [Building constants during setup saves 452 first-network requests with byte-identical predictions](building-constants-during-setup-saves-452-first-network-requests/README.md) | 2 | adopted |
| 2026-10-16 | [Compact-layout frames bound lazily: slow setup 4.611 to 4.585 seconds for 809,972,800 first-network FLOPs](compact-layout-frames-bound-lazily-slow-setup-4-611/README.md) | 2 | adopted |
| 2026-10-16 | [Twenty-network prebind census refit: 836 fewer first and 295 fewer warm requests](twenty-network-prebind-census-refit-836-fewer-first-295/README.md) | 1 | adopted |
| 2026-10-16 | [Full census refit makes the block-greedy join release pass: request risk 23.10 to 17.21%](full-census-refit-makes-block-greedy-join-release-pass/README.md) | 6 | adopted |
| 2026-10-16 | [Rank-384 stack census refit: 977 fewer first and 351 fewer warm requests](rank-384-stack-census-refit-977-fewer-first-351/README.md) | 1 | adopted |
| 2026-10-16 | [Candidate-specific prebind refit saves 603 cold and 495 warm requests under slow setup, with identical bytes](candidate-specific-prebind-refit-saves-603-cold-495-warm/README.md) | 4 | adopted |
| 2026-10-17 | [Two exact setup offloads: terminal Smirnov scale batch (-8/-9 RPC per circuit) and 64 main-read aliases in setup](two-exact-setup-offloads-terminal-smirnov-scale-batch-8/README.md) | 11 | adopted |
| 2026-10-17 | [Concurrent official startup reproduced a setup failure hidden by isolated qualification](concurrent-official-startup-reproduced-setup-failure-hidden-isolated/README.md) | 1 | adopted |
| 2026-10-18 | [Usage-ranked prebind list reduced requests by 167–438 with byte-identical heavy-column-sketch predictions](usage-ranked-prebind-list-reduced-requests-167438-byte-identical/README.md) | 2 | adopted |
| 2026-10-18 | [Compact setup repair removed duplicate handles and met a strengthened 4.20 s startup bound](compact-setup-repair-removed-duplicate-handles-met-strengthened-4/README.md) | 1 | adopted |
