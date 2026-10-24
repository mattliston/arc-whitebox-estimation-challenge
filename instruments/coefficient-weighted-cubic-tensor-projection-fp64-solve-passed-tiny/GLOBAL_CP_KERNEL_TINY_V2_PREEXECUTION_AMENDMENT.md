# Tiny v2 amendment, before any execution

September 7, 2026 PDT. The v1 test source is preserved UNEXECUTED. Independent
source review identified a current-bank diagonal convention not mirrored
there: the CP reader overwrites its matrix-product diagonal with a separate
FP64 cube sum from actual stored FP32 factors and weights.

Run qualify_global_cp_kernel_tiny_v2.py, which adds that overwrite for BOTH
old and replacement planes, with its actual cost in the restoration stage.
It preserves both fixed fixtures, selections, solve law and every numerical
threshold. It does not alter the bank, response reader, model or targets.

V2 additionally seals its source and all three prospective notes before
and after execution; records interpreter, Python, FNP and NumPy versions;
requires the FNP0.12 series; and preserves environment/nice failures inside
its result. Its synthetic distinct-error diagnostic denominator is clearly
named as the full old tensor norm, including star. The existing query check
uses all 5^3 cross-contractions of the five fixed vectors, stronger than five
individual scalar cubes. None of these changes follow a numerical result:
neither v1 nor v2 has been executed at this amendment.

Expected isolated native charges at n8/a6/M12/K6: three ordinary FP32
products 2376 FLOPs; one FP64 LU solve 432 FLOPs. The whole tiny circuit
includes other work and is recorded separately, not presented as a P2 bill.
The response-row caveat remains: no masked-reader/head/rollout invariance
is established by this synthetic certificate.
