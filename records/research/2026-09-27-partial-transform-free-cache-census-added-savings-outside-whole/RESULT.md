# Partial arithmetic-block reuse: insufficient saving

The fixed4096-scalar arithmetic-block cache fails the prospective1.5%
complete-bill ceiling gate on both unchanged S31 TRAIN networks.

| TRAIN rank | Free arithmetic-block reuse | Added saving outside whole-input repeats | Union with earlier free whole-input cache |
| --- | ---: | ---: | ---: |
| 35 | 0.326720% | 0.001047% | 0.346794% |
| 100 | 0.359323% | 0.000460% | 0.381678% |

The union is a descriptive, optimistic combination without double counting;
it does not change either failed gate. All lookup, copy, storage and lifetime
costs were set to zero. Neither number is a deployable saving. No cache
implementation follows, and no block-size or other parameter is selected.
The retained thin-star component remains below the submission bar.

The test observes only paid additions/subtractions inside the two existing
global A/B input-transform bindings. Exact repeated ordered operand blocks
produce identical output hashes. Existing free resident reuse contributes
no arithmetic. This closes this fixed partition's cache rationale, not all
partitions, scalar common-subexpressions, output transforms or new algebra.

All16 prediction files and complete bills equal the independent controls;
both transform sequences, shapes and individual transform bills exactly
match the prior whole-input census. Twelve scalar-byte fixtures, exhaustive
tiny duplicate grouping, signed-zero/operation distinction and an in-place
case pass. Reconciled operation sums agree with all network/transform receipts.
The catalog itself was held in memory; full operand/hash records were not
persisted, so the receipt audit is not an independent full-data regrouping.

The first attempt failed only its synthetic strided-buffer qualification,
before native inputs. Its version and guard remain preserved; REPAIR.md
explains the exact-block repair. The repaired guard completed and retired
in212.52s, peak RSS6.352GB, zero added swap.
Exec session69990 was reaped with exit0. No targets, training, VALID40,
new samples, changed estimator, package or upload were used.

S31 #332517 remains2.1736465056908115e-9 adjusted. This test provides no
banked gain. Standing submission authorization and the overall goal persist.
