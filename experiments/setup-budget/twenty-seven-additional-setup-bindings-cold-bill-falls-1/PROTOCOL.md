# S32 fixed 27-key setup extension

Source-only follow-up on exact S32 `hill_climb_star_projection32_v1/candidate.py`,
SHA256 `f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef`.
No model, target, numerical unit, worker or setup execution is authorized by this
source-preparation receipt. Root controls any later qualification.

The sole candidate change appends `fragment.py`, prepending exactly its fixed
27 keys to `_PB_SINGLE`. These keys were identified from the existing exact
rank35 billed census and were absent from the incumbent's final 600-key tuple.
No other source, kernel, arena, layer, coefficient, cache capacity or wall guard
changes. Removing the appended bytes recovers the parent exactly. `build.json`
pins the parent, candidate, fragment, builder, this protocol and census receipts.

## Cache lifecycle and composition

The final `_pb_bind` (parent line7748) binds the same 24 KS accumulator views and
calls the existing two-phase LVF binder (7020). Its phase1 executes `_ipt_side`
for `_PB_SINGLE`; phase2 fills each entry's fused operation tuple. Existing
`_ipt_side` keys are `(L,rows,cols)` within separate A/B/C caches. Each of the
27 additions is unique and absent from the 600-key list. Final per-kind initial
counts are A194/B252/C181, below unchanged capacities320/352/320.

Fresh complete setup therefore changes624 bound/600 fused to651/627. The
four-second internal guard still governs both phases; completion is an observed
qualification requirement, not guaranteed by the source. Entries share the
same arena; their extra views/operation tuples add handles. Eviction uses the
unchanged LRU policy; an evicted entry is rebuilt normally. `_pb_reset` clears
the same caches, and setup's native/RPC arena-loss handling remains unchanged.

The retained nine D3 keys are disjoint (all nine have depth3; this set has
depth4–6). Composing both fragments gives660 bound/636 fused when complete.
The UVP B-cache changes no setup keys or cache capacities; it remains compatible
with this append. Neither composition is built here or claimed qualified.
Prebinding can change LRU order and the amount of later binding, so component
bill deltas must not be added without the actual composed receipt.

## Cold and warm arithmetic

For a new tree key `(kind,L,r,c)`, let `a=r/2^L`, `b=c/2^L`.
Parent `_ipt_view` (5519–5524) reshapes a tree of `7^L*a*b` elements.
Parent `_lvf_rs_level1` (6885–6906) additionally reshapes
`4^(L-1)*7*a*b` elements. The ordinary FP32 reshape tariff charges one FLOP
per element; remaining binding/fusion operations create views and do not change
the arithmetic of the prediction. The build receipt records both terms per key.
The 27 primary tree reshapes total1,384,690,314F (0.5007023% of the rank35 bill);
the second term is additional and is calculated explicitly in `build.json`.

When all27 keys are actually first bound during the parent rank35 prediction,
complete candidate setup moves both terms into setup. This is the predicted
cold-bill difference under no intervening eviction; actual bill must be checked.
It is not eliminated arithmetic or a worker-average saving. On a warm call
where both sources already retain all used keys, their binding-bill difference
is zero; existing fusion may still reduce request count. Different networks may
use only a subset or cause different evictions. The fixed two-rank sequence and
eventual worker panel, not the rank35 estimate, determine amortized economics.

## Proposed bounded qualification, root only

First run the existing native evaluator on35,100 under a180-second guard.
Require all16 FP64 prediction bytes equal S32, each bill no larger than S32
(a warm tie is allowed), and mean paired bill reduction at least0.1%.
Stop on a failed native gate. Otherwise reuse the existing worker/boundary/setup
harnesses, in that order, under serial guards with180/240/120-second limits,
respectively, RSS below7.5GB, zero added
swap, complete retirement, exclusive receipts and a candidate hash check before
each job. Use ranks35,100 in the fixed worker2 and boundary2 sequence and three
fresh setup trials. Compare all16 promoted FP64 prediction arrays byte-for-byte
against S32 native TRAIN2. Do not require unchanged cold FLOPs; report measured
cold/warm deltas against the exact derivation and compare boundary FLOPs to the
worker/native receipts. Require full651/627 setup counts and all three setup
times at most4.1s. The usual handle ceiling remains strictly below70,000.

Compare boundary CPU, wall, request count and participant residual to the saved
matched S32 receipts; do not infer runtime benefit from FLOP transfer. Require
mean paired server CPU ratio at most1.04 and the conservative residual proxy
`0.285450751 + 4*max(0, maximum paired residual delta) <= 0.35`.
If parity,
setup or resource qualification fails, preserve the failure and stop. Passing
this small probe supports only an exact component; any worker20 composition,
broader score claim or submission requires root's subsequent admission. No
standalone score gain,0.5% worker-average gain or new bank is claimed here.
