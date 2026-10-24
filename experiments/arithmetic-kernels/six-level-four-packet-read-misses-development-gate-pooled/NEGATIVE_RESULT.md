# Read6 four-packet variant — closed at the fixed development gate

This fixed variant is not retained or submitted. Both individual improvements
and a pooled adjusted ratio<=0.99 were required. Actual pooled ratio is
**0.9950934019424302**, and rank100 is slightly worse.
No broader panel, real-client run, package, padding/packet/penalty search or
amplitude/precision tuning follows this failure.

| Rank | Weight5 raw | Read6 raw | Weight5 cold FLOPs | Read6 cold FLOPs | Adjusted ratio |
|---|---:|---:|---:|---:|---:|
|35|1.87125543957e-08|1.8548489395e-08|405547750335|404792021179|0.989385217466|
|100|1.85468233405e-08|1.85923440386e-08|408575923244|407905682722|1.0008099093|

The small full-cold savings are much smaller than the isolated large-product
savings. New cold view bindings are charged, and only shapes divisible by64
can select D6. Changed FP32 arithmetic slightly changes final predictions;
it is not an accuracy-preserving cache optimization like Weight5. No physical
or learned parameter was changed. First-layer outputs are exactly unchanged,
35 repeats byte-for-byte, and warm/cold100 outputs match exactly.

## Positive kernel evidence retained with its limited scope

Independent fullD6 arithmetic matches all three tested shapes byte-for-byte:
1024x8192x1024,832x8192x768,64x1088x64. The reference separately allocates each
original frame, keeping each allocation below the4GB array cap. Its largest
complete reference workspace is4728649728 bytes, not one allocation.
Packet product bills match the exact rational fullD6 formula; interval lifetimes
cover parent leaves, both child levels, dead-buffer reuse and all four parent
backward stages. Interleaved repeat passes. Maximum FP64-direct relativeRMS
is7.2189531068e-5, below the fixed3e-4 tolerance.

The two large warm bill ratios toD5 are0.9361604986 and0.9514411279. Warm
wall time is materially worse (about0.624 versus0.406s, and0.554 versus0.290s).
Small geometry dispatches ordinary multiplication. These unit facts do not
establish a useful whole-estimator improvement or production runtime.

The new arena is1568602112 bytes,398297088 above Weight5/Read5. Unit peak
observed RSS5423808512 bytes. Full development peak observed RSS6937296896 bytes,
zero additional swap throughout; all owned groups retired under the unchanged
8GB guard. Four full predictions incurred1591065723250 billed FLOPs;
this excludes separate kernel/reference/scoring research arithmetic. All four
outputs and bills were sealed before opening the existing35/1001B target means.
This is reused development evidence, not independent validation.

Candidate SHA-256: `5c61cb81b298eefc6125d39f91084c4dc2d8ca7403b9637fe43e19d816ea0c52`.
Parent Weight5 SHA-256: `a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9`.
Complete parent prefix is preserved; builder/kernel/source/receipts are retained.

Weight5 remains qualified at local existing14 NC3.883207214854186e-9; its exact
776949-byte archive and pending approval question are unchanged. Official
#331584 remains4.015535762022727e-9, so no official4e-9 milestone has yet been
banked. Goal~1e-9 stays active and unfinished. No upload was attempted here.
