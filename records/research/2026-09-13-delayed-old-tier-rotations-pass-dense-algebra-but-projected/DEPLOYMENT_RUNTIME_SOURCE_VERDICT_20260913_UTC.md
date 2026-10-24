# Source verdict: retain tiny PASS, no45s deployment claim

Only source and sealed scalar JSON were read. No Python, numerical imports,
arrays, candidate materialization, target read or runtime test was performed.
The positive dense tiny V2 remains valid; V1 and BOX's runtime objective remain
negative. No full-card lazy runtime run is selected from this evidence.

## Conditional complete count, not just a FLOP percentage

Replayed the fixed L8/10/12/14 lazy and L9/11/13 consolidation schedule against
all40 original current-c2 geometries. Each diagnostic SHA was checked against
native40_v2/result95d80ab15860bd51865c828a48d09f195642fa6ac6d8f86f9c5e99b4f864d83e.
Rank37 diagnostic SHA is
b2eaf2e9d5ccad81ebdf08ef49b6bc265d0fc31a5ccf19550d6550a0d3b804ba.
The actual BOX source378092f7 supplies its final square-depth tables,
_shared_old384_mm row split, _mm output/contracted8192 tiles, cached kernels,
and two-box D6 dispatch. No source function or estimator was executed.

For a cached depth-d product, arithmetic calls are
(add,subtract,stack,copyto,matmul)=(7d,8d,2d,d,1), plus its empty output.
A covered BOX D6 product uses(70,80,20,10,3), plus its empty output.
Output tiles pay a concatenate; contracted tiles pay every partial-result add.
All additional finite/abs/max/cast/typed-scalar/guard/concat operations and the
corrected21F envelope subtotal are included. Basic slices are0F but still RPC
bookkeeping; zero-F unstack cache misses are history-dependent, not assumed away.

| Conditional rank37 change | Budget calls | FLOPs |
| --- | ---: | ---: |
| add |+640|-144,318,612|
| subtract |+728|-103,648,560|
| stack |+182|-63,054,504|
| copyto |+91|+27,625,140|
| matmul leaves |+21|-6,435,592,380|
| concatenate |−2|+29,097,984|
| all other changed operations |+141|+44,802,103|
| Total, excluding frame-cache misses |+1,801|−6,645,088,829|

Net rank37 kernel inventory changes by −1 cached D4, +13 cached D5 and
+3 BOX D6 dispatches. This is more primitive calls, not fusion of the parent
call graph. Four large factor rotations disappear, but split reconstruction,
split reads, map application and consolidation introduce additional products.

Across40, mean non-cache count change is+1,900.85 (range+1,801..+2,097).
Mean complete-F change is exactly−6,615,094,645.75, reconciling the corrected
source price. Initialization remains3,286,988,570F, with no new persistent
binding or arena. Different FP32 predictions could change subsequent geometry;
these are conditional counts, not an observed future complete bill. A full
operation-count prediction also needs the resulting128-frame cache history.

## Data movement and actual time evidence

Rank37's add/sub scalar work falls247,967,172F, only0.814334% of PROFILE's
30,450,303,322F in those categories, while their call count grows4.808605%.
For FP32 add/sub, count two reads and one write per output; for stack/copy/
concat, one read and one write. Under this logical-element convention, their
net traffic falls3,026,257,104 bytes. Extra finite and abs/max scans add about
243,818,500 byte visits. This is not measured DRAM traffic: cache reuse,
strides, BLAS buffering and matrix-kernel reads remain unmeasured.
The existing source liveness allowance remains an additional78,184,448 bytes
at the recorded maximum old-bank width; the same1,168,900,096-byte arena stays.

Actual PROFILE result ed1cd38a7bd0822e233295bd06ddce6b303ccfc07ef6f2be1fbc7a50be3b55fc
has49,590 operations, add/sub backend19.871371251s, matmul7.658592069s and
reported RPC overhead11.761010530s. It has no per-shape/block times. Most of
the lazy FLOP saving is in matmul, not the measured add/sub bottleneck; the
changed non-cache calls alone add3.631781% relative to that observed count.
Neither the tiny's0.323s nor a FLOP-proportional extrapolation establishes
full-card speed. There is no credible measured path here to saving6.22900175s
from BOX's51.22900175s while adding these dispatches.

## Decision and missing evidence

No full-card run follows this component-only positive. Any later packed
qualification must measure the COMPLETE altered block, not time only the
omitted rotation. A narrowly scoped candidate would include one fixed L14
join and its L14/L15 consumers (recorded R16384,j1344,s544/512,n1024,r384),
both eager/lazy paths, the actual original arena/packed schedules, setup,
all guards/copies and independent fidelity. It would qualify that component,
not45s or raw score; no such test is selected here.
To overturn the runtime conclusion, evidence must show that the removed
blocks save at least6.229s AFTER all added blocks/dispatch/setup costs. Current
rollups do not provide that attribution. No closed prefix-view, depth,
MICRO3, box-size or rank experiment is reopened by this note.
