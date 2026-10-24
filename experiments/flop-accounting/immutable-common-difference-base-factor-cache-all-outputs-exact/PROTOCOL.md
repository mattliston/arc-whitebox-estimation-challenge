# Fixed three-slot UVP base B-transform cache

Parent: exact S32 `hill_climb_star_projection32_v1/candidate.py`, SHA256
`f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef`.
No birth-count candidate or unbanked D3 prebinding is included.

This is one exact implementation component, not a fit or new approximation.
Cache only the join10 materialized rank320 F generation, continued as the
identical lazy base at join13. Original Fu/Fv split arithmetic and all
reconstruction, read, padding, packet, leaf, and unwind arithmetic remain.
Only already-computed B transforms can be reused. Three D6 slots are eligible:
Fu first4096 columns, Fv first4096 columns, Fv remaining padded columns.
All inner transposes, recent-join factors, other ranks/generations, and
unsupported ancestry run the original forward transform.

The cache strongly owns the original F and its exact blocks/ru metadata.
Actual slice expressions carry source offset and valid/stored widths.
Actual original zero-and-copy padding carries ancestry only after the copy.
No callback ordinal, numerical hash, bare object ID, shape-only association,
or unmetered array operation can grant a hit. The key includes generation,
leg, offset, valid width, depth, padded K/N and dtype. A live product holds at
most16 ancestry records; overflow disables reuse for that product.

Each saved tree is one ordinary counted `fnp.copy` outside the arena.
The leaf consumes it read-only. A native copy may change storage strides or
BLAS packing; complete all16 output-byte parity is decisive, not merely the
mathematical equality of the saved transform. Exactly three entries and512,000,000
persistent bytes maximum; excess/unsupported shapes simply do not cache.
Replacement and prediction-finally clear all array ownership. Existing arena,
thread settings, setup and numerical cache caps are unchanged.

Saved S32 rank35 geometry predicts336,946,736 retained bytes and about
319,701,476 net saved FLOPs (0.11560%), after one paid copy per slot.
This is a source estimate using historical exact D6 transform charges;
current census has no owner/byte hashes and is not measured cache performance.

Root reviews source and owns all execution. First run fixed native ranks35
and100 against fresh/matched exact S32 controls, serial guard7.5 GB RSS,
zero added swap. Require all16 FP64 output bytes exactly equal, complete
paid bills and cache/copy/retirement counters recorded, and mean full-bill
improvement at least0.1%. Preserve any failed result; no tuning to endpoints.
Then qualify resource/client-boundary behavior before retaining as a bundle
component. The submission policy remains fully validated aggregate gain>=1%.
No truth arrays, learned parameters, new samples, score-dependent selection,
board/API calls or submission are part of this source preparation.
