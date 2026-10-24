# Whole-product reuse has zero opportunity on the two current traces

All **295 outermost _mm products** on each of networks35 and100 have distinct
canonical operand-pair fingerprints. Including transposed equivalents does
not merge any classes. The two zero-operand products are already-free empty
products at layer0/1. Consequently the most favorable exact/transpose/zero
whole-product reuse ceiling is **0 FLOPs, 0% of either complete cold bill**.
The fixed5% implementation gate fails, and no cache candidate is created.

| Network | Complete cold FLOPs | Outermost _mm FLOPs | Classes, including transposes | Optimistic removable FLOPs |
| --- | ---: | ---: | ---: | ---: |
| 35 | 413448696909 | 381662313492 | 295 | 0 |
| 100 | 416264792412 | 384507304163 | 295 | 0 |

Both observed forwards reproduce all16 original output bytes and complete
cold bills. Every one of the295 ordered shape/dtype/caller/per-call bill
records matches the previous qualified current-bank census. The _mm wrapper
was restored, original meter and arithmetic were unchanged, and native
caches retired. No targets or reference arrays were read.

Streaming hashes match independent one-shot serialization for contiguous,
Fortran, transposed, strided, reversed, empty and signed-zero arrays. Three
nonfinite fixtures are rejected. Synthetic product groups match independent
pairwise enumeration. An additive audit strengthens the first unit's
preservation check with BEFORE-call copies, read-only arrays and five real
multi-block fixtures; all pass. Independent exhaustive pairwise comparison
certifies all native groups and zero-product costs. Signed-zero bits remain
part of identity, except for completely zero operands.

This closes only whole-product identity/transpose/zero reuse on these two
traces. It does not cover partial transforms, intermediate subproducts,
matrix-chain reassociation, low-rank or approximate structure, direct
non-_mm operations, other inputs or other algorithms. Research hashing time
is not deployment runtime. No production cache or changed cost is claimed.
The two unweighted Gram identities are already inside _resident_gram512 at
layer0; their full combined bill is only859975168 FLOPs (about0.208%/0.207%).
That post-result observation selects no further numerical study.

Five guarded groups retired after two full calls totaling
829713489321 FLOPs. Peak sampled group RSS was
6317850624 bytes; self-reported native peak was
6433816576 bytes. Zero added swap. About
16.860 GB was read through
1MiB payload blocks; no wide arrays or dataset copies were saved. No fitting,
samples, GPU, downloads, candidate or upload.

Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,
100/100 complete,zero failures. The4e-9 milestone is banked;3/2/~1e-9 remain
unmet. The full goal is active. The qualified0.41%-better local composition
and its exact archive are unchanged. Its prior automatic upload-review
rejection still awaits the already requested approval; this turn made no
upload attempt or alternate external write.
