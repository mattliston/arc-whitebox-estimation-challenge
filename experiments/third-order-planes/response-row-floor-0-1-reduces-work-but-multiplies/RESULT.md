# Fixed response.1/P1 endpoint: rejected on the actual objective

2026-09-09. Two TRAIN/development networks35/100, fixed existing1e9
truthpost references. This is neither independent validation nor a board
result. No fitting, threshold selection or further candidate was run.

| Role | Mean raw final MSE | Mean complete FLOPs | Mean per-net adjusted |
|---|---:|---:|---:|
| d50 |1.88209325453344e-8|470235568710|4.024874313854929e-9|
| P1 |1.8762623409467065e-8|470235568710|4.01240370696628e-9|
| response.1 with frozen P1 |1.0356745998995807e-7|415869748378|1.958783714424208e-8|

Candidate/d50 raw ratio5.5027804675, adjusted ratio4.8666953591;
individual adjusted ratios4.3306885504/5.4242758251, zero wins.
It misses the predeclared5% mean adjusted gain and neither-net-worse gate.
No full validation or submission. The complete-cost saving is real,
but the fixed-head candidate is not an improvement. This does not by
itself close all possible matched response-only designs.

Native35/100/repeat35 integrity passed: all16 domains,47 declared floors,
exact repeat and source-owned startup. Actual startup1383913786 FLOPs
is included in the cold score. Independent resource watch122.965328s,
maximum sampled combined RSS6694223872 bytes, zero swap growth,
maximum interval.250778s, owned process retired. These instrumented
native timings are NOT a production residual/wall certificate.

Build04e91f1b178451e3d20c576afd904cb49311688be23ce4140fbb9eebdec5e835.
Candidate9d32027aedbb5554896e2b85144d68eba13d06e8599bc70445b589ab88f0a4bd.
Natived815ba128b23b5e1e9ddee4e885a2945628ccf01e60c58fe6169b8eca821cf63.
Watch20cd87ab3fd7ed8711500d5ea83f53bbfb7fd397659ab3480488bff6ae72dc56.
Scoreafa75665bedf4d4ccc9e51b13209a822cdbf54df9369760f033a17131b0722a1.
Score completion332363a89c814c993bae00f9c7450f1d9662fddc32c8c81cff1ceec3dc388842.
Scorer17.061976s, peak303562752 bytes. All predictions and own bills sealed
before reference access. Raw computation uses float64 differences to
float64 targets from the actual saved float32 outputs.

The first scorer failed target-free on an erroneousfloat64 output guard,
receiptce4364aafa13c6277bfea4f5f9abda2207f44fac17db26d2a128ec777b4ce897.
It was preserved, and the separate source-inverse dtype correction is
documented in SCORE_DTYPE_V2.md. No estimator was rerun to obtain the score.
