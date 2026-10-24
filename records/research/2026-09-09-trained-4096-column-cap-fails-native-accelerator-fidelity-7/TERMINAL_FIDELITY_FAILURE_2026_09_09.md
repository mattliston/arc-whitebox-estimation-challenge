# Terminal cap4096 fidelity failure — no score or advancement

2026-09-09 UTC (2026-09-08 Pacific). Frozen six-epoch/240-update training completed, SHA256 `5499433f54ed648c66716cfd9044fa7908c6024f9bb20012f6c091d56f8500d8`. Its terminal candidate SHA256 is `604d23a2118cfd7781ff3a51b4adc9d9125c586b051a9cc797c23a0573735a6a`, build receipt `027186a362bb487a76f993b7c9b9a9c68c4be1157f7f05cd1aa8e403219ff4a1`.

Native40 plus repeat completed with its own official-input bills, result SHA256 `ad5107badc63f81630595db4f5d6058f1ee44cc2c6be44b3af38147308f1c598`. Mean complete40 compute309,716,647,338.85F, initialization1,383,913,666F included once. Maximum41 wall38.1984921s, RSS4,191,535,104 bytes. These native gates passed, but do not alone establish training-shadow compatibility.

The preset final judge failed BEFORE hashing or opening any VALID truth, at the native/MPS all16 maximum-mean-difference<=1e-5 gate. Failed result SHA256 `7f07b494eed79a512e716cfdd3a4134fb1845ec29bba242899056b843730abce`; passed=false, completed=false, truth_accessed=false. No raw or adjusted score, no panel advancement, no submission. All receipts/source versions remain intact.

A subsequent target-free comparison of the stored output arrays, with no estimator execution, found7/40 failures:

| Rank | Maximum all16 difference | Layer of maximum | Final-layer RMS difference |
| --- | ---: | ---: | ---: |
|67|3.4943223e-5|10|4.0146398e-6|
|139|3.1381845e-5|13|2.6531084e-6|
|42|1.9729137e-5|11|2.1719376e-6|
|181|1.7940998e-5|12|2.9298266e-6|
|234|1.7657876e-5|13|2.4871226e-6|
|169|1.5363097e-5|12|1.8741939e-6|
|39|1.2993813e-5|9|1.7984215e-6|

The other33 networks remained below1e-5. Rank67 is below2.15e-6 through layer9, then jumps at layer10. Its current layer10 retirement occurs after recording mean10, so that retirement cannot explain the first jump; retirement9 or selections10 are the first relevant discrete events to inspect.

Independent source audit found no payload/transfer defect. Native shared-newborn pushes and Torch dense-factor pushes use different FP32 operation order before FP64 pair-energy ranking; hard cutoff divergence is a concrete hypothesis, not an observed cause. Original-payload35/100 qualification does not certify all trained-payload membership cells. Neither terminal native40 nor the saved terminal validation outputs preserve those membership diagnostics.

Cheapest unresolved diagnostic, not executed here: fixed rank67, unchanged terminal source/payload, target-free native and MPS forward observation, recording physical retirement and gate identities plus margins; both observed runs must reproduce their own already-sealed output bytes. No forced selections, tolerance relaxation, VALID truth access, score, refit or rank/epoch sweep is admitted by this note. No such run was launched as part of this failure record.

This failure is isolated from the successful submitted d50 bank #330390 (official adjusted4.53886349950395e-9).
