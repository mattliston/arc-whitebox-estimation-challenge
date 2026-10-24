# Two fixed frontier capacity boxes - source draft

This is additive source preparation, not a candidate build or numerical result.
No model, arrays, targets, Python preflight, fit, stage or numerical experiment
has been executed by the author. Root owns the independent watched RPC test.
All previous sources and positive/negative receipts remain unchanged. There is
no depth, packet, shape, precision, rank or physics search and no automatic
advancement. The official current-c2 submission's repeated TIME_EXHAUSTED
failure remains a failure; a local component pass cannot rescue that receipt.

## Exact policy and source boundary

Parent is the complete current gain-slice c2 source
`../current_comp_shared_gain_slice_v1/build_v1/estimator.py`, SHA
`c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea`.
The exact tested frontier owner is `../current_comp_push_frontier4_v1/kernel.py`,
SHA `16e121a0f557bd1705ca5c620e1125ed157b463c1421f5bfad8b162d8fba2c46`,
with its protocol SHA
`56321ed32ce1927ff0530e9e1e62ebddeee9d970551422a0a09c9aa2b7400256`.
Its actual source ancestor `../current_comp_push_microbatch3_v1/kernel.py`, SHA
`90abc9bcd8ef543c5e3d4306b540af0a72a7210f0333e30e869e52a3b258a6fb`,
is retained as an exact source owner, not relabeled a successful timing arm.

Inside the original last-effective `_packed_product_4` broad PUSH guard:

1. If `k == 384`, use capacity box A `(1024,384,8192)`.
2. Otherwise, if `n <= 6144`, use box B `(1024,1024,6144)`.
3. Otherwise, execute the original general serial-D6 PUSH unchanged.

The entire original guard remains byte-exact: two-dimensional FP32 operands
and output, `m=1024`, `0<k<=1024`, `1024<n<=8192`, k/n divisible by64, matching
inner/output shapes, and `arena_views is _SCRATCH[1]`. No READ, square,
nonaligned, empty, oversized or alternative-arena path is newly intercepted.
The old final D4 fallback, all tiling and every square schedule remain exact.
In particular, box A is NOT widened to `k<=384`; first-match ordering is fixed.
Unknown future `k!=384,n>6144` geometries retain the original serial fallback.

`compose.derive_source(tick=None)` performs two disjoint positional edits:
the original guarded serial return acquires those two branches, and the sole
original paid `_p1_d6_832_bind(_SCRATCH[0])` statement is immediately followed
by `_fbx_bind(_SCRATCH[0])`. Every original binding is still performed. One
ASCII block of literal view specifications and helper definitions is appended.
Removing that block and reversing both edits must restore the ENTIRE original
UTF8 source byte-for-byte. Learned literals, gain slicing, all physical heads,
the shared384 constructor and all consumers are unchanged. Last-effective
original kernel, binder, `_mm` and shared push/read/commit spans are separately
compared bytewise. No numerical work is added to module import.

## Tested arithmetic and actual-shape views

Both boxes use the identical tested four breadth levels, then the exact three
depth-two packets `(0,800),(800,1600),(1600,2401)`, then four upper unwinds.
The tested child/frame definitions are restored by whole-span inverses after
only private renaming; allowed child lengths remain exactly800 or801. The
tested upper body is restored by its whole-span inverse after only renaming
and replacing fixed frame/group keys with actual `(m,k,n)` keys. Its numerical
operations, sources, signs, copies and operation association are untouched.
The tested owner itself retains its exact original-c2 child/upper inverses.

The two new leading seven axes of each child precede the packet axis, which
preserves the old flattened prefix ordering. Each logical D6 leaf uses its
ACTUAL dimensions `(m/64,k/64) @ (k/64,n/64)`, not the capacity dimensions.
No padded arithmetic, leaf copy, gather, transpose or extra reconstruction is
introduced. Same scalar expression does not guarantee backend byte identity;
full-candidate actual bytes remain a mandatory measured gate.

At cold binding, each capacity's full contiguous nested A4/B4/P4 arrays also
gets one paid flat alias. Their shapes are `(2401,mcap/16,kcap/16)`,
`(2401,kcap/16,ncap/16)` and `(2401,mcap/16,ncap/16)`. Runtime makes ONLY basic
packet-and-matrix slices of those aliases. The equivalent nested and flat
offset for a prefix index t and matrix coordinate `(i,j)` is
`base + t*rows_cap*columns_cap + i*columns_cap + j`.
This identity still holds with narrowed actual rows/columns. The code never
reshapes a narrowed padded-stride array: doing so could copy or join holes.
Child frames slice the actual root `(m/16,k/16,n/16)` and batch800/801 from
fixed801-lane capacity views. The unused801st lane is never consumed by an800
packet. Cache keys include capacity-view identity, actual dimensions, depth
and child batch; different geometries cannot reuse an incompatible frame.

## Complete live-value proof and ownership

The original arena is unchanged:292,225,024 FP32 elements /1,168,900,096 bytes.
There is no new backing arena. Both new binding sets and every old set alias
that SAME arena; handles alone do not make their old numerical contents live.
Inputs and output remain the original caller's separately owned arrays.

Let `q=7/4`, `H_j=(mk+kn)q^j`, `P_j=mn q^j`, at capacity dimensions.

| Capacity box | H4 | P4 | Child region | Total elements | Bound view F |
| --- | ---: | ---: | ---: | ---: | ---: |
| A:1024,384,8192 |33,191,424|78,675,968|160,763,904|272,631,296|649,085,920|
| B:1024,1024,6144 |68,841,472|59,006,976|130,620,672|258,469,120|692,932,352|

Peak used bytes are1,090,525,184 and1,033,876,480, each strictly inside the
original1,168,900,096-byte arena. These are capacity bounds, not measured RSS.
The unchanged full estimator, inputs, outputs, remote handles and other live
state must still pass the independently measured full-tree8GB ceiling.

Upper A/B buffers alternate until A4/B4 occupy `[0,H4)`. P4 occupies
`[H4,H4+P4)` throughout all three packets. The reusable child region begins
at `H4+P4`. For child root `(cm,ck,cn)=(m,k,n)/16`, write
`L=49*(cm/4)*(cn/4)` and
`AB2=49*((cm/4)*(ck/4)+(ck/4)*(cn/4))` per lane.
The fixed child high water is `801*max(2*L,L+AB2)`. At box A the unwind
`L+P1+S2=2L` dominates. At box B the leaf plus still-live AB2 dominates;
incorrectly assuming2L for box B would understate memory.

A1/B1 uses the not-yet-written leaf region; A2/B2 is disjoint above the leaf.
A1/B1 is dead before matmul writes the leaf. A2/B2 is dead before its range is
reused by P1/S2; the leaf is dead before S1 reuses it. Child output is the
separate actual packet slice of P4. All completed/pending P4 packets remain
conservatively protected while the next child reuses its region.
Only after ALL children finish may upper reconstruction reuse old A4/B4 and
child ranges. Each unwind retains disjoint incoming Pj, destination P(j-1)
and scratch Sj. Final output is outside the arena.

`layout_metadata` checks all27 complete view intervals and23 conservative
live phases for EACH fixed box:4 upper-forward +3*5 child +4 upper-unwind.
Every simultaneous interval pair must be disjoint, and the maximum stop must
equal the reported high water. Box A's specs, regions and phases must also
equal the actual tested owner's metadata exactly. Actual smaller views are
subsets of these conservative ranges, so that capacity proof applies without
an input-dependent optimizer or retained-dead-value assumption.

One shared `_FBX_BUSY` flag spans either box's entire operation and is cleared
in `finally`; entry also rejects either old serial binding being busy.
There is no `_mm` or estimator callback inside the new arithmetic, only the
same synchronous FNP operations. The original participant is serial: no
concurrent call, nested scratch consumer, asynchronous outstanding request or
numerical profiler callback is permitted. This is not a new general-purpose
thread-safe guarantee. Each binding retains at most128 upper frame/group
entries; the existing global128-frame cache remains bounded. Caches hold only
basic arena views, never caller inputs/output. Final qualification must retire
the persistent client AND server groups; no fake zero-RSS or early arena
release is inferred from dropping a Python reference. A future in-process
native harness must explicitly clear both new binding frame/group owners as
well as the old caches before making any weak-reference retirement claim.

## Full price, not a multiply-only saving

At actual covered `(m,k,n)`, complete depth-six numerical F remains

`7^6*(m/64)*(n/64)*(2*k/64-1)
 + (7/4)*(mk+kn+mn)*sum_{j=0..5}(7/4)^j`.

This includes every forward add/subtract, identity stack/copy, leaf matmul and
output reconstruction. No nonzero or zero copy is removed. Max-shape F is
3,421,951,456 for A and6,483,751,680 for B. Warm numerical calls become
`(add,subtract,stack,copyto,matmul)=(70,80,20,10,3)`,183 rather than655.
The old serial's seven children become three depth-two packets after a shared
upper frontier; this is not a change in scalar F. First use of a previously
uncached shape requires an upper16-unstack frame plus two child8-unstack
frames,32 total. Real cache eviction/history can change zero-F counts; all
actual operations must be recorded, not forced equal to an old opcount.

Each box pays27 reshape calls. All aliases count, including the duplicate
flat aliases. The binding tariff is

`sum(H1..H4) + sum(P1..P4) + (3/7)*sum(P1..P4)`

`+ H4+P4 +801*[sum(child H1,H2)+sum(child P1,P2)+(3/7)*sum(child P1,P2)]`.

For A this is303,480,320+111,867,392+233,738,208=649,085,920F.
For B this is319,221,760+127,848,448+245,862,144=692,932,352F.
Thus EXACT extra cold F is1,342,018,272, with54 new paid reshape operations.
Both original serial PUSH278,566,656 and dormant READ282,489,344 binding F
remain paid, as do all existing D4 bindings, square startup, heads and inputs.
There is no credit for an unused old binding or for a repeated call's setup.

Expected current rank37 own bills, subject to actual measurement:

- Cold:406,399,683,439+1,342,018,272 = **407,741,701,711F**.
- Repeat: **404,454,713,141F**, unchanged complete warm F.
- Difference / own startup:1,944,970,298+1,342,018,272 = **3,286,988,570F**.

Original array-input production billing and native preparation conventions
remain separate. Full-call F must include its actual input/init/output work;
no constant is subtracted to make a served bill appear cheaper. The saved
persistent40 mean would gain33,550,456.8F from one extra cold initialization,
but process-per-MLP production must pay its actual cold cost instead. No such
amortized projection qualifies an inference bill or objective.

## Existing measured evidence and narrow full-card test

Actual fixed-shape FRONT judge result/completion/finalization hashes:

`e604ea1de105484ce637a8b1112c9943e995d41f471bd24983277d28adc848e2`

`19e87fb88e0c52124b77b6661c17c0ff84009d30f3112955e14264b9b940775e`

`87135226790783318840de4299814ae1574010269104041882bd33471fb27e51`.

Its actual watch result/completion/finalization:

`f9bc136078ac5e3b5b80563dec9dcf890ed7022e7e65567638b60f886c03c6f7`

`0961bef1cb124f15f33954f5d74af0c5c3ecd862f7a5628bea240986e26774dc`

`b4f136992707bebca426170f716adf34f5c2842ff3ee987498d43c9b11c0ffab`.

All outputs and complete warm F were byte/exact, with fresh warm maxima
serial.557365042s, micro3.506108917s, frontier.373902958s. Ratios were
.6708403466753482/.7387796291287236, passing that component's fixed gates.
This does not prove another shape's speed or whole-card45s. Old micro3's
5.5% speed result remains below its10% advancement gate. Full-card D4's
changed bytes and failed45s are also preserved, not promoted as an exact fix.

Saved target-free current-c2 native40 metadata source is
`../current_comp_shared_gain_slice_v1/native40_v2/result.json`, SHA
`95d80ab15860bd51865c828a48d09f195642fa6ac6d8f86f9c5e99b4f864d83e`,
with positive watch result
`1bc1135f275609a4ea64bd7414de2cae9881f9396af3b561f6af6e1fe2bc7bfb`.
Excluding repeat, actual1128 D6 PUSH dispatches comprise798 at k384 and330
at k768/832/896/960/1024,n<=6144. All1128 fit the two fixed boxes; mean28.2
calls/net, range24..31. Their mean87,159,741,594.275F is21.6648% of own full
mean402,309,518,125.6F. Rank37 has24 calls,20 in box A and4 in B; ten calls
are exactly the tested full A shape. This census is not a runtime selector.
No saved per-dispatch wall measurement exists: the fixed-shape speed cannot
be multiplied into a whole-card speed claim or45-second guarantee.

The selected next falsifier is ONE current-candidate real production RPC
cold37/repeat37 pair using the original canonical input and unchanged
NumPy-free client under root's separate `watch_rpc37.py/RPC37_PROTOCOL.md`.
Require all16 output arrays byte-identical to saved exact c2, own repeat byte
identity, original finite/nonnegative checks, full own cold/warm F and exact
startup above. Advancement requires BOTH new walls<=45s and new cold wall /
the actual saved c2 cold wall<=.80; the latter is explicitly historical, not
a fresh paired comparison. The old62.916841667s c2 cold was over its60s gate
and its negative seals remain negative. No baseline retry is authorized.

Original actual context<60s/residual<.4s, whole300s, closed-call progress120s,
full-tree8GB, zero added swap, actual observations<=1s, one-second final seal
tail and complete owned-group retirement all remain independently enforced.
This RPC endpoint does NOT observe internal all47 selectors/all16 unfloored
domains, native parity, targets, raw score or full40/panel. Those cannot be
claimed from source algebra or endpoint bytes. Same outputs with higher cold
F do not imply an adjusted-objective gain. Any later changed-source scoring
and release needs its own complete costs and explicit authorization.

## Source API and preflight boundary

`compose.authenticate(tick=None)` returns flat direct source pins;
`compose.derive_source(tick=None)` returns `(full_text, JSON_native_proof)`.
Required proof keys are `full_source_inverse_verified`, `parent_sha256`,
`candidate_sha256`, `source_bindings`, planned cold/warm/init F and nested
`kernel_proof`. There is no writer or numerical run flag. The derived SHA is
computed only by this inert source operation, never a fabricated build seal.
`kernel.source_and_proof(tick=None)` returns `(ASCII_appendix,proof)`;
`kernel.layout_metadata(capacity)` and `kernel.price(m,k,n)` are target-free
stdlib arithmetic only. Helpers authenticate sources before loading them;
no source read/hash/AST work occurs in a paid numerical product.

Root must read all three new sources/documents, then preflight the actual
source and watcher entry/parser/API before any build or run. This draft has
not been executed by its author. Root's actual future source qualification
and numerical result must be separately recorded; no success is inferred
from the old fixed-shape frontier receipt alone.
