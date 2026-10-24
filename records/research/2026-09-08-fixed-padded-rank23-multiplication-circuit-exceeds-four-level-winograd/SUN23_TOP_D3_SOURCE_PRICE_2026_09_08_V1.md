# New rank23 source: one fixed top-level circuit fails its price screen

September 8, 2026 PDT. Phase 2 only. Primary-source reading and exact
integer arithmetic, no model arrays, targets, numerical kernel, candidate,
fit, current-trajectory census or submission. Earlier closed FMM circuits
and their numerical/runtime failures remain unchanged.

## New source and bounded question

[Sun, arXiv2604.27645v1, April30,2026](https://arxiv.org/html/2604.27645)
prints a general-ring 3-by3 rank23 multiplication scheme with13/13/30
left/right/output additions. Its scalar model treats copies and sign
changes as free. The author supplies a public integer verifier. This is
new to the bounded inspected catalog inventory, not a new rank record or
a measured FNP saving. Root read the printed straight-line program.
The separately suggested DPS2026 rank48 paper was already in the old
mixed-FMM envelope and was not reconsidered as a new experiment.

The one specified circuit checked here replaces the TOP level of existing
D4 by this Sun23 level, leaving three packed Winograd levels below each
of its23 products. Pad each dimension up to a multiple of24, so every
3-way child is divisible by8. No placement, depth, leaf-size, precision,
orientation or shape-policy optimization is performed.

The motivation is algebraic exactness with a nominal2.6455% reduction in
the cubic leaf term versus D4. Expected actual raw change is UNKNOWN
until finite-precision/current-head qualification. No whole-card gain or
runtime improvement is predicted from the multiplication rank alone.
Cheapest decisive prerequisite: compare the lower arithmetic subtotal of
this literal fixed circuit against two archived dominant D4 shapes. If it
already loses with packing/sign/setup costs free, do not implement it.

## Exact integer calculation and scope

The existing packed Winograd stage charges four additions and three
element copies on each input, and seven output additions. For dimensions
divisible by2^d its FP32 recurrence is

    M0(m,k,n)=m*n*(2*k-1),
    Md(m,k,n)=7*M(d-1)(m/2,k/2,n/2)+7*(mk+kn+mn)/4.

This includes the bank's input packing copies and all inner-product
subtraction terms. It exactly reproduces the archived D4 prices below.
Let (a,b,c)=(8*ceil(m/24),8*ceil(k/24),8*ceil(n/24)). The new circuit's
FAVORABLE lower subtotal, before its own top-level copies and signs, is

    23*M3(a,b,c)+13*a*b+13*b*c+30*a*c.

| Archived input shape m,k,n | New padded shape | Existing D4 F | New lower subtotal F | New / D4 |
| --- | --- | ---: | ---: | ---: |
|1024,1024,4096|1032,1032,4104|5,180,428,288|5,200,316,284|1.0038390640492156|
|1024,832,8192|1032,840,8208|8,417,538,304|8,466,479,579|1.0058142028265844|

Thus this specified circuit is already .384%/.581% MORE expensive before
its missing top-level packing, signs, padding input copies, extraction,
allocation, casts, guards and dispatch. Those extra charges cannot create
a saving. These are two archived same-shape comparisons, not a new current
409f complete bill, all-shape theorem or a final-score result.

Necessary fully batched final leaf A/B/C storage alone is522409580B and
889279636B respectively. This omits all earlier transforms, scratch and
outputs, and is NOT a complete arena or memory qualification. No timing
or residual-wall claim follows from that lower storage count.

## Decision

Do not implement or numerically run this top-Sun23/three-Winograd circuit.
Its source-price prerequisite failed. This does not prove every use of
the published algorithm inferior, but no alternate placement/depth sweep
is selected merely to rescue the first negative comparison. A different
implementation would need its own credible complete-cost premise and
runtime/accuracy obligations before any numeric work.

The bounded positive-result inventory found no measured exact gain left
uncombined with qualified409f. Two unimplemented micro-edits remain only
source-priced: denominator reuse58748956F and dead final covariance
assignment11535360F. Use the corrected HELD_MICRO_COMPONENT_TARIFF_CORRECTION
receipt, not the earlier overcounts. Their combined projection is only
.0150806% of the current mean bill; preserve them for an otherwise justified
combined change, not a standalone qualification or milestone submission.

Confirmed official #329926 remains4.982231460183304e-9. Qualified local14
409f remains4.770235434670105e-9. No raw/adjusted gain, upload, forum post,
commit, promotion, deletion, frozen edit or closed-arm reopening occurred.
