# Centered fourth-moment channel: small stack component

Recommended component is `candidate_95c_centered22_HELD_v2.py`, SHA `7b203c5ee8c948a472fc9348efbfd03800ebe86162c1e56c8615745d1d43927c`. It is built and syntax checked, **not native validated or banked**. Its four source-line substitutions preserve all source line counts and binding assertions.

The pair map is bilinear in its Hermite derivative vectors. Changing only channel 22 from `bb` to `bb - (2*m/s)*aa` incorporates its three mean corrections before the dense bilinear map. The resulting postprocess is `pk22_centered - 2*cov*cov`. Channels covariance and K21 remain untouched. This is exact over real arithmetic. The actual active grouped Hermite routine on synthetic 32-by-32 inputs agrees to 2.50e-16 relative in binary64 and 1.16e-7 in binary32; covariance and K21 are bit-identical. The production radial recurrence may nevertheless amplify changed binary32 rounding and needs paired native evaluation.

Local public flopscope **0.12.3+np2.2.6** prices the old 1024-by-1024 postprocess at 14,162,430 FLOPs and the new vector centering plus postprocess at 7,378,942. For fourteen layers this saves **94,968,832 FLOPs**, averaging **0.0395428%** over the saved current-source science32 bills. At unchanged raw MSE that projects from **1.44685656e-9 to 1.44628443e-9**. No claim of an accuracy improvement is made.

The numeric-call estimate is only two fewer per layer, about 28 total: six dense operations are removed, four vector operations added. Version 2 retains all four new vector outputs in the existing E2 hold so three intermediate handles do not cause extra unbatched frees. This is not a solution to the roughly 800-call residual shortfall. There are no new persistent arrays, setup kernel families or prebind entries.

A direct scalar contraction still pays arbitrary-plane multiplication and reductions, and discarding the expanded third plane requires additional gathers and weighted packed reductions. No credible path to the earlier gross 0.171% ceiling survives those replacement costs; this component's measured 0.0395% is the realizable result found here. Peak probe RSS 335MB; no full model, boundary server, native, judge, forum, credentials or fire was run/read.

Artifacts: `PROBE_CENTERED_CHANNEL_v1.json`, `CENTERED22_PROJECTED_PRICE_v1.json`, `BUILD_CENTERED22_v2.json`, probe and versioned builders. Original v1 exists for provenance but v2 is recommended for batch handle lifetime.
