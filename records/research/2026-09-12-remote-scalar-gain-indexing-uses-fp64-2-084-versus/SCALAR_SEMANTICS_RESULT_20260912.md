# Actual scalar interface diagnostic, September 12, 2026

The target-free v3 diagnostic passed in both actual interfaces and its external
supervisor exited zero. No estimator was imported or executed; no truth, fit,
candidate edit, raw-error measurement or release qualification occurred.

The original candidate remains 4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3.
Only its 16 gain constants were AST-decoded. The actual paid FP32 gain bytes
hash to df273d1e2c227c33cdf915a0e849ba799ce19fef7bfb2c2cd83beafca266f21b.
Both modes used the same binary-exact deterministic length-1024 vector.

For all 15 active layers, native indexed gain arithmetic charged 3,090 F;
remote indexed arithmetic charged 2,084 F. Native tanh/factor/vector were
FP32, whereas remote tanh/factor/vector were FP64. The native FP64 hook cast
cost 2,048 F; the already-FP64 remote hook view cost zero. Thus the measured
1,006-F per-layer difference sums to exactly the independently observed
15,090-F complete network discrepancy. Original indexed output bytes differ.

The one-element-slice control charged 3,090 F in **both** interfaces, and all
six recorded intermediates were byte-identical for every layer. Complete
tiny contexts were native 94,780 F and remote 79,690 F, including their own
2,080-F preparation. Every operation and snapshot-budget check reconciles.
This confirms the prospective scalar mechanism and an expression-level
control; it does not prove that this is the sole cause of full-network drift.

Whole supervised time 2.074482542462647 s; sampled combined tree peak
172,933,120 bytes; maximum completed-observation interval 0.21692641731351614 s;
zero added swap; all owned client/server and observational groups retired.

Worker e6336483af82f75364efe64a95eaa0855a676c19ceaf878e5d74d9908580afca;
protocol 9ff60bc1ab796df0e9911119a0f0bc24a1ee063226a21f6c6bea0e5dec489f6f;
watch 42f643f2d484f39faa52060d36f7983b13c59b083f764df36899245f4a26f297;
watch specification 691da41946775190b1652f40f047dcaffc267c6a823328d390a59b4832171c12.

Actual result/completion/finalization hashes, respectively:

- Local: e1025386a0a6ab5be11630ed59b9b1abeef35b53b0d7b84667c55fc34a62d9c0;
  7791c0631e9e8e049a74f6700642ab15f8d8805eb4dd5cee7f0010bbf76c4101;
  b5a91fcc95cdf93bed70908cf4ec46f3f7f5248626886f9ab4e2bb40b67f6f6a.
- Remote: 9bf16e7ebf2f8a6373f86ca8526340e94418a2ef1ee164613874829ab29d0d13;
  82c8bf07648beab195a4fa9a97a65b6b5d9ca874664b07e73692cf261f1fd194;
  742428c039b8c818fd776803223e1cc1cde24e9bdeb34fc7186243d081db8a6a.
- Watch: 31e484037eb7a4cb2965be9ebb16d5a8ee749c305cc240876b48c71e8d672541;
  fc1ca2aaeb5e8a09a2621469f3a4194cb6e538f0d8843ee154ae04fc4e81389d;
  73a76214e4a0a5ecf35e68c28962656e05f4417bfdfdf1f3bd4d11e969c1f1b4.

V1 metadata-preflight failure, v2 remote snapshot-API failure and every old
negative estimator fidelity/backend receipt remain unchanged and negative.
Next: a separately versioned gain-slice candidate, with no parameter change
or refit, must pass actual full-network native/server comparisons and its
own complete objective/resource/release checks. No old bill or score is
silently transferred to that new candidate.
