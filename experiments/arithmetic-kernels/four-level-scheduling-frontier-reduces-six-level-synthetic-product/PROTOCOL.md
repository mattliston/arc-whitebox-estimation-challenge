# Fixed breadth-four / depth-two-packet PUSH — source draft

Source preparation only. Neither this document nor the inert kernel authorizes
a numerical run, estimator change, target access, fit, build, panel, score,
submission, or automatic next experiment. Root must fully review the source,
perform an actual metadata/source/API preflight, and separately select a watched
test. Existing sources and all positive and negative receipts stay unchanged.

## Fixed question and prospective gate

For exactly `(m,k,n)=(1024,384,8192)`, retain the current c2 six-level Winograd
scalar tree but put the serial boundary after FOUR breadth-batched levels.
Their 2,401 independent products are `(64,24) @ (24,512)`. Execute the remaining
depth-two subtrees in exactly three prefix packets **800,800,801**, then unwind
the four upper levels once. No shape, depth, packet, layout, precision, backend,
seed or numerical-policy sweep. This is a runtime component, not a raw route.

The prospective four-role order is `d4, serial_d6, microbatch3, frontier4`, each
with fresh local IPC server/client ownership, one paid setup and three paid
products (cold0,warm1,warm2), with separately billed output materialization.
All use the original micro3 fixed synthetic fixture and independent FP64
reference convention. No existing graph, model, weight, truth or saved model
array is required. Each role's repetitions must be byte-exact, and frontier4
must match serialD6 and micro3 outputs byte-for-byte. Complete warm F must be
identical. D4 is an informational runtime/error floor, not a proposed replacement.

Timing usefulness is the conjunction

`max(frontier warm1,warm2)/max(serial warm1,warm2) <= .85`

and

`max(frontier warm1,warm2)/max(micro3 warm1,warm2) <= .90`.

Ratios require these fresh controls. Historical numbers may be descriptive but
are never called paired measurements. Operational completion and this timing
Boolean remain separate. Failure closes this fixed arm without tuning/retry.
Whole runtime must remain300s, one numerical tree below8GB, zero added swap,
completed observations at most1s apart,120s retired-context progress and hard
worker/server lifetimes, and the original one-second final publication tail.
Each actual context retains the original `<60s`, residual`<=.4s` and full-bill
guards. The new independent supervisor/worker are NOT part of this source task.

## Source and arithmetic identity

Exact c2 source:

`../current_comp_shared_gain_slice_v1/build_v1/estimator.py`

SHA `c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea`.

Only the authenticated metadata/kernel owner
`../current_comp_push_microbatch3_v1/kernel.py`, SHA
`90abc9bcd8ef543c5e3d4306b540af0a72a7210f0333e30e869e52a3b258a6fb`
is loaded as a stdlib-only source owner; it is never a candidate import.

The three controls are exactly that owner's source-derived kernels and layouts.
The frontier child is a unique whole-span inverse of its batch-aware child:
only names and allowed packet lengths change. That owner reruns the original
current-c2 frame/child inverses. The upper body is a unique whole-span inverse
of the original cached product: its four forward/unwind levels are unchanged;
only the single leaf matmul becomes three calls to the exact depth-two child.
No Winograd product, addition, subtraction, identity copy, source sign or
reassembly association is changed or omitted. No arithmetic is performed at
import. `make_namespace(fnp)` compiles only selected authenticated kernel spans.

The new axes at each recursion level precede the old axes. The four existing
prefix axes flatten in their current contiguous order, never by permutation:
`(7,7,7,7,64,24) -> (2401,64,24)`, and similarly for B and P. The child's two
new seven axes precede the packet axis. Every logical leaf remains
`(16,6) @ (6,128)`; every corresponding leaf and ancestor follows the original
scalar operation order. Packetting does not create cross-prefix arithmetic.
Actual BLAS batching/strides can still affect floating execution, so bit-exact
equality is a mandatory measured gate, not a guarantee substituted for a test.

## Exact arena / lifetime proof

The caller's original arena remains **292,225,024 FP32 elements /
1,168,900,096 bytes**. No second arena, operand clone, padded numerical block,
gather, leaf copy or enlarged scratch allocation is introduced. Inputs and the
unchanged33,554,432-byte output are caller-owned and separately reported.

Let `q=7/4`, `H_j=(mk+kn)q^j`, and `P_j=mn q^j`. At the frontier:

| Simultaneously retained region | FP32 elements |
| --- | ---: |
| Both upper level4 operands, H4 | 33,191,424 |
| Complete upper level4 output frontier, P4 | 78,675,968 |
| One reusable801-lane depth2 region | 160,763,904 |
| Total high water | 272,631,296 |

That is **1,090,525,184 bytes**, leaving78,374,912 bytes of the original arena.
The proof conservatively regards every completed/pending P4 packet as live;
it does not overwrite old packet outputs to manufacture headroom.

Upper A/B levels ping-pong until level4 resides in `[0,H4)`. P4 starts at H4;
child storage starts at `H4+P4`. Each child lane needs200,704 elements:
leaf P2=100,352, disjoint P1=57,344 and S2=43,008. During descent, A1/B1=24,192
elements occupy the not-yet-written leaf region; A2/B2=42,336 occupy the future
P1/S2 region. A1/B1 die before the leaf write; A2/B2 die after matmul and before
P1/S2 reuse. During the last child unwind, the dead leaf region supplies S1.
The output is the disjoint packet slice of P4. All three packets use the same
child storage serially. The800 packets never read the unused801st lane.

Only AFTER all three children finish do upper A4/B4 and all child numerical
values die. Upper reconstruction may then overlap their former ranges; each
step keeps its entire incoming Pj disjoint from destination P(j-1) and Sj.
The final destination is the caller's output. `layout_metadata()` enumerates
all23 conservative phases and rejects any physical interval overlap between
simultaneously live named values. All27 bound view intervals are also checked
against the original arena and exact declared shape products.

Views/frames remain handles, not live numerical values, after their last
consumer. Only arena views, never fixture/input/output references, are cached.
A busy flag spans upper descent, all children and upper reconstruction and is
cleared in `finally`. No other `_SCRATCH` consumer may run while busy; no
reentrancy, concurrent kernel, callback numerical work, retained result alias
or in-flight server request is permitted. Before releasing a role, clear both
frame/group dictionaries, the namespace frame cache and all arena view owners;
the inherited weakref/whole-process retirement proof must still pass.

## Complete cost, including setup

The unchanged complete depth-d arithmetic/copy tariff is

`7^d (m/2^d)(n/2^d)(2k/2^d-1)
 + (7/4)(mk+kn+mn) sum_{j=0}^{d-1}(7/4)^j`.

It includes all transforms, literal identity copies, leaf matmuls and output
reassembly, not merely nonzero multiplications. All four controls/new role
include actual input loads, arena/output allocation, every paid reshape and
separate paid output materialization. No callback or cold work is subtracted.

| Role | Warm F | Numerical calls | Setup F | Cold unstack calls |
| --- | ---: | ---: | ---: | ---: |
| d4 | 3,930,963,456 | 73 | 1,068,953,600 | 16 |
| serial_d6 | 3,421,951,456 | 655 | 1,630,009,600 | 24 |
| microbatch3 | 3,421,951,456 | 291 | 2,112,169,728 | 44 |
| frontier4 | 3,421,951,456 | 183 | 2,000,528,864 | 32 |

Frontier numerical counts `(add,subtract,stack,copyto,matmul)` are
`(70,80,20,10,3)`: `4*18+3*(2*18+1)=183`. The32 first-product unstack calls
come from one upper frame16 plus child frames8 each for batch800 and801.
Other zero-F metadata operations remain actually reported, never rebased.

Exact new binding sum:

`sum(H1..H4) + sum(P1..P4) + (3/7)sum(P1..P4)`

`+ H4+P4 + 801*(66,528+157,696+67,584)`

`=303,480,320 +111,867,392 +233,738,208 =649,085,920 F`.

The middle term charges all three explicit nested/flat alias reshapes. The
original cached helper's unused dependency-list slots are literal `None`, not
hidden new views. Total setup retains original D4 PUSH+READ1,068,953,600 and
dormant D6 READ282,489,344, replacing only this isolated role's serial PUSH
278,566,656 with649,085,920: **+370,519,264 cold F**. There are109 setup
reshape operations and two zero-F empty allocations; original input loads
add no operation rows. Complete setup+first product F is5,422,480,320.

This is NOT a full-card cost quote. If a later fixed-shape-only estimator
insertion retains the general serial PUSH binding for other shapes, it must
pay the ENTIRE649,085,920 additional cold F. No such insertion is selected.
All heads, predictions, selectors, global startup and full-estimator resources
would need their own new qualification. No old all16/all47 proof is inherited.

## Prior evidence and negative scopes

Micro3's actual fresh result is numerical/operational PASS but usefulness FAIL:
warm maxima serial`.588333833s`, micro`.555942583s`, ratio`.944944097750027`,
above its`.90` gate; D4`.139227166s`. Judge result SHA
`8d7cba82f071fc3ea4c984e9c7cfc8c60c6707dba5a6c0716c54148bd3d8faa3`;
watch result SHA
`fb2a8bb9e020dd0d1e7357808d80f3d94383a1a92c5d4d23134fb9cd5ddbebf9`.
The new proposal reduces upper-transform dispatch repetitions, not another
claim that micro3 passed. Full breadth D6 and descendants already batch all
seven axes; simply naming tensor-axis batching would not be new.

`streamed_leaf_layout_v1.py` and `streamed_leaf_winograd_v1.py` instead retain
five breadth levels, then stream the seven final products and remove identity
copies. Their old `.29512 -> .28113s` PUSH result is a different final-level
schedule, already closed, not this middle-prefix cut or a speed prediction.
The old shape-tight D4 and extra leaf-copy failures remain closed. This layout
uses exact fixed dimensions solely to exhibit a legal lifetime plan, not as
new evidence that compact strides alone improve speed.

The broader scheduling principle HAS been tried in the different rank40
Smirnov circuit: `SMIRNOV_GLOBAL_BATCH_REPRESENTATIVE_V1_CLOSED.md`, SHA
`271a8a3ac38b945ec337ff92df846f425c25a556ec9c41db870354e99ea99f9b`,
records3,623->1,083 calls, exact same arithmetic and fitted memory, yet8.096x
D4 warm wall. Its backend alone exceeded its threshold. That negative remains
negative; fewer calls and a valid static layout do not guarantee runtime.

Current c2 has59,803 warm rank37 ledger operations; many operations outside its
24 serial-D6 dispatches are already breadth-batched D4/square transforms, not
unfused physical scalars. The current rectangular kernels already use pooled
`out=`. Public F71/F80 pooling evidence is real for that author's unpooled
backbone, but public F79 also measured two elementwise fusions ineffective.
No current source/receipt establishes a large whole-card elementwise-fusion
gain. This bounded fixed-shape test is the one selected information question.

## Inert API handoff

`authenticate(tick=None)` returns flat direct source pins. The compatible API
is `source_bundle()`, `child_source(bundle)`, `make_namespace(fnp)`,
`setup(fnp,ns,literals,role,left,right)`, `product(ns,role,left,right,state)`,
`price(role)` and `setup_price(literals,role)`; `ROLES`, `COUNTS`,
`SETUP_RESHAPE_CALLS`, `COLD_UNSTACK_CALLS` are explicit. State retains
`arena/output/state`, and the state owner exposes `.frames/.groups/.busy`.
`layout_metadata()` is stdlib-only shape/cost/liveness algebra.
`source_proof()` returns `(literals,proof)`, exactly the proof of
`make_namespace(fnp)`; the proof is JSON-native so a saved/read-back receipt
compares exactly without tuple/list schema aliases. The historical top-inverse claim is nested ONLY in
`control_kernel_proof` with its three explicit control roles, never asserted
for the new frontier schedule. `setup_operation_counts(role)` and
`cold_extra_operation_counts(role)` return exact per-operation count maps;
`initial_operation_counts(role)` combines `COUNTS` with first-product unstack.
Numerical product calls use the already-admitted metadata owner; they do not
rehash/read the large source inside each paid context. Full source rehashing
belongs to admission and finalization. All new numerical views are still
created and billed inside actual setup, not source/proof construction.

This kernel has no run/help CLI and creates no output at import. No worker,
watcher, candidate or output stage is authored here. All source and integer
shape proofs remain DRAFT until root's independent complete readback and
source-only API preflight; the author has executed no Python or numerical work.
