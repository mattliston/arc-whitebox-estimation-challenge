# Fixed four-by-four-bit popcount product: source-price rejection

September 8, 2026 PDT. Phase 2 only. Root and the existing efficiency agent
read installed FNP.12 sources. No estimator, numerical kernel, array
acquisition, precision experiment, fit or changed candidate was executed.
The two different circuits rejected in LOW_BIT_AND_JOINT_LAW_ADMISSION_AUDIT
remain closed; this is not a rescue of either one.

## Concrete question and prospective gate

Consider a fixed four-bit-by-four-bit matrix product represented as sixteen
binary bitplane pairs. Pack inner coordinates into uint64 words; for each
pair, broadcast AND, count set bits and sum over words. Bit significance,
signed offsets and quantization scales then reconstruct the approximate
real product. Test the supported API and dominant operation bill FIRST.
If this fixed core is already more expensive on the representative dominant
CP shapes, do not run quantization or pursue a bit-width sweep to rescue it.

Actual raw quality was UNKNOWN before this screen and remains unknown.
No signed-pair cancellation, head compatibility or whole-card accuracy is
implied by exact integer arithmetic after quantization. No expected final
raw gain was asserted. The possible motivation was only a sufficiently
large complete cost reduction to compensate for any measured raw increase.

## Supported operation, unfavorable real price

FNP exposes bitwise_count; installed NumPy metadata is 2.4.6, above the
wrapper's minimum 2.1. It is a counted integer ufunc, not a host callback.
The installed default operation weights for AND and popcount are both one,
but the resolved uint64 dtype rate is TWO. In particular, popcount returning
uint8 does not lower the rate of the uint64 operand it actually reads.

For q=ceil(k/64), each of the m*n* q packed-word entries therefore pays:

- uint64 bitwise_and: 2 F;
- uint64 bitwise_count: 2 F;
- word summation: q-1 F per output with an explicitly bounded 32-bit
  accumulator. Default uint64 accumulation is more expensive.

The sixteen-pair core subtotal is consequently

    16*m*n*(5*q-1).

This already grants every other operation free, including bit significance
and accumulation BETWEEN the sixteen pairs. A 32-bit word-count accumulator
is arithmetically sufficient for the two stated inner dimensions; no
overflow or dtype underbilling is being assumed as a discount.

| Archived shape m,k,n | Fixed popcount core | Existing D4 shape price |
| --- | ---: | ---: |
| 1024,1024,4096 | 5,301,600,256 F | 5,180,428,288 F |
| 1024,832,8192 | 8,589,934,592 F | 8,417,538,304 F |

These are archived authenticated same-shape comparisons, not newly acquired
current409f trajectories or complete candidate bills. The core is already
about2.34%/2.05% more expensive on these representative large CP pushes.
They do not establish an all-shape theorem. In particular, no old measured
push fraction was silently transplanted into a new whole-card score.

The complete program would further pay input ranges, quantization, signed
corrections, bit extraction/packing, bit weights, casts, accumulation,
rescaling, padding, copies, chunking and all unchanged card work. Installed
packbits charges input elements; unpackbits charges expanded output
elements, at their applicable rates. No reinterpretation/packing shortcut
was implemented or assumed qualified. Added costs cannot reverse the
negative margins of the specified representative products.

## Decision and source pins

Reject this specified fixed circuit at its price prerequisite. No numerical
kernel, precision test, fewer-bit rescue, fit, current-trajectory recapture
or complete-card run follows. This is not a universal low-bit multiplication
closeout and says nothing measured about quantization raw error.

SHA256 under data_gate/.venv/lib/python3.11/site-packages/flopscope:

- numpy/__init__.py:
  33edacfc6ca01866d4f558e3a43f8fadfd6a2c0efdbc4e4712acc1d839ad4e2f
- _pointwise.py:
  7d376515446f1b174bef375c48116a552098bd514f5552450876f56822beb0c7
- _array_ops.py:
  e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c
- _accumulation/_reduction.py:
  a60b2ef74d568c1a1db5ad0cc06edcdea438efa0fdec7830b17bf439bdf65536
- data/default_weights.json:
  f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a

Current qualified local409f remains4.770235434670105e-9, not a new board
score. Frozen candidates, packages, receipts and the completed response-row
exception are unchanged. The below1e-9 official goal remains active.
