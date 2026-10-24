# Complete numerical/billing pass, timing failure: do not advance

Completed2026-09-09 02:22:08 UTC, September8 Pacific. Phase2 only.
Fixed seeded1024x8192 by8192x1024 synthetic product, one cold/two warm calls
per role, real NumPy-free FlopScope-client.12 and the bank's OpenBLAS server.
Each role used a separate client/server lifetime. No MLP/target or score.

| Role | Warm productF | Maximum warm context wall | FP64-reference RMS error | Sampled combined peakRSS |
| --- | ---: | ---: | ---: | ---: |
| CurrentD4 |10409197568|0.295003708s|1.9912512466005153e-5|1440022528B|
| BreadthD6 |8833219584|0.735729750s|7.023140158240704e-5|3565387776B|
| Serial-topD6 |8833219584|0.943101167s|7.023140158240704e-5|987529216B|

All repeat bytes agreed within each role. SerialD6 and breadthD6 were exactly
byte-identical. All expected primitive counts and warm formulas passed.
Serial's extra setup was exactly561056000F, including both new max-capacity
view sets inside the existing arena. Reference errors are against a separate
offline ordinary-FP64 product of the same FP32 input bytes, not neuron means.

Serial maximum warm wall divided byD4 is3.1969129249046593, versus the
prospective ceiling1.35. Its maximum residual0.004388582s is below0.20s.
Thus the timing gate FAILS even though numerical, billing, memory and residual
checks pass. Runtime receipt `passed=true, completed=true` means successful
completion of this measurement; `single_product_qualification_passed=false`
and `timing_gate_passed=false` explicitly deny advancement.

The smaller working owner did not buy throughput here. Warm serial backend
time0.7350–0.7641s is already above breadthD6's0.6967–0.6993s; serial overhead
0.1583–0.1748s versus breadth0.0339–0.0380s compounds it. These are actual
fixed-product client timings, not estimates of a full estimator or Linux worker.
CurrentD4 warm backend0.2697–0.2729s and overhead0.0209–0.0216s are faster.

Decision: no complete candidate build, layer/shape/depth sweep, schedule tuning
or submission follows this result. Preserve the9.5–9.7% hypothetical whole-card
FLOP opportunity and this implemented arithmetic/memory component, but its
measured wall trade does not qualify it for deployment. The official bank
remains330390 at4.53886349950395e-9 adjusted. Strict official<1e-9 is unmet.

Seals:

- Kernel:a8d1cb1a972b37afd9eaaa67acf820cba3196dabea10ba1058bb3409c0b20ec7
- Client:d1cb99f30149e5b4350db4df81983fdbea0fb0099be88a9015de3fb267b27899
- Driver:961ab14418c109b0e85ffb665795fb83f50dbce21a90c3cc63dadde4d6c7aabc
- Protocol:888674b8ebfb4fe3f0e1caa071d9528d69c45c880967ffbbb827167536212477
- Launch contract:510183b2fc3283d51f8b75d68fc2c3b3a18c728d7dfb3c7f547e118f8e78d474
- Result:86ce7e8b1ea3d9dcc81c42bad6cb78828b520f38aed5f4918acf6052a6cf8660

Root read all source and verified the helper/layout/top-DAG source audit before
execution. Independent accuracy review identified two pre-run harness issues:
the client.12 constructor has no wall-limit keyword, and role limits needed a
final post-cleanup check. Both were fixed before execution without changing
any threshold; final source hashes were cleared. All artifacts are preserved.
