# Preserved first qualification failure

Fixed qualification v1 failed on September9,2026 PDT, after10.5945s worker /
11.5531s external supervision. Zero full-size calls and zero retired-suite
markers. The worker reached the tiny MPS scalar readout after its dual
trajectory, then Tensor/Python-float division raised TypeError. No training,
target read, raw-error or adjusted-score measurement occurred.

CPU tiny reverse/JVP scalar check passed: absolute discrepancy2.53753e-13
on a1.47932e-7 directional scalar. Independent stored-plane repair error
2.35814e-18; unrounded projection defect1.89735e-18. These partial results
do not make the qualification PASS. External peak combined RSS670,777,344
bytes, max interval0.331717s, swap unchanged736.75MiB, owned groups retired.

- Worker result848f31f81456e16656c66f679af2d50fdc80f19373314e7b51f1d9d4575efd58
- Worker completion46de520189cc382dabdbdbf4cf035082a84e0c1c28eb5eb3e71810750d44b8be
- Worker finalizationb161710482bd7ad216a5973955d5f9268f3704361fe52b545d9e4b02d9913799
- Watch result45749012206ade20b15feea664fac900d1c900e83f4d9e4eff3d38ad628d787a
- Watch completion3564887d566b1e2c458162b41bce984d7c709e92c2ac890f61c626daa29c3558
- Watch finalization1acb1250adb4b735840c125ce9363f34d35e8d0a5c12749d8145eead4c85180a

A separate fixed8-element CPU/MPS primitive diagnostic reproduced the
MPS Python-float division failure and showed that same-dtype tensor division
and reciprocal multiplication are supported but may differ by one primal
ulp. It ran no model:1.17686s,316,981,248 bytes RSS,27,754,496 driver bytes.
This identifies a library dispatch compatibility issue, not failed geometry
value or mathematical-gradient evidence. Exact internal backend cause is
not proven. The original source/operator and all failed receipts remain
frozen. Any supported vector-JVP diagnostic must be separately versioned,
fully qualified and retain the same scientific/numerical/resource gates.
