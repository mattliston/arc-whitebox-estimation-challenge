# Common-cubic extension: fixed CHECK4 closeout

Recorded 2026-09-10 UTC after the sealed score and its independent supervisor
completed. Phase 2 only. Operational PASS; scientific gate FAIL. This arm is
closed: no scale/layer/LR sweep, refit, P co-fit rescue, native40 expansion,
independent14, production run, or submission is advanced by this result.

## Paired final-score evidence

Four prospectively fixed networks, ranks 37, 39, 42, 48; original one-billion
sample truth files, stored float32 and widened to float64 for scoring. Native
predictions and complete own bills were sealed before truth access. Cold 37,
warm 39/42/48; the exact repeat of 37 is a lifecycle check, excluded from means.
The objective is the mean of each network's raw MSE times its own scoring
multiplier, not the product of aggregate means.

| Configuration | Mean final raw MSE | Mean complete FLOPs | Mean adjusted |
| --- | ---: | ---: | ---: |
| Frozen current COMP | 1.9933680058595472e-8 | 475126242316.25 | 4.2964338254612944e-9 |
| Fixed terminal CC | 1.9958315243117663e-8 | 488236228163.75 | 4.420548267828534e-9 |

CC/COMP ratios: raw **1.0012358573253797**, complete compute
**1.0275926368191926**, adjusted **1.0288877816834323**. Raw wins 2/4;
adjusted wins **0/4**. Neither the primary <=0.95 adjusted / >=3 wins gate nor
the smaller strict-gain / >=3 wins component gate passed. The nominal 4.421e-9
must not be compared against the official bank's different network panel.

| Rank | COMP adjusted | CC adjusted | CC/COMP adjusted |
| --- | ---: | ---: | ---: |
| 37 | 4.51604278831124e-9 | 4.632823669716281e-9 | 1.025859117567996 |
| 39 | 3.798008499495254e-9 | 3.929228143501814e-9 | 1.034549591983272 |
| 42 | 4.6492506142600086e-9 | 4.793391393402326e-9 | 1.0310030134104222 |
| 48 | 4.222433399778674e-9 | 4.326749864693714e-9 | 1.0247052955105242 |

## What this establishes

The selected A1-preserving common-cubic control, inside the existing two signed
factor slots at births 10..14, did not earn its complete 2.7593% cost increase.
It changed raw error by only +0.1236%. This is the fixed 65-parameter fit described
in DECISION.md and TRAIN_PROTOCOL.md, not an oracle ceiling or a statement that
all possible state representations are exhausted. Original P/A/cstar remained
frozen; endogenous H and downstream state dependence remained live. The full
stored-float32 coefficient defects, including nonzero delta-A1, and all required
plane/diagonal updates were represented. No target-derived current-layer input
was used.

## Operational and numerical checks

Qualification, training, zero build, terminal build, native-zero3,
native-terminal CHECK4 plus repeat, and scoring all passed their independent
supervisors. Original source/preflight drafts and receipts are preserved.
Null CPU/MPS checks were exact; native zero matched COMP at all 16 outputs and
47 observations. The terminal cold/warm repeated network was exact. All native
bill observations included startup, including the cold Q getter; no nested
startup double-counting was introduced.

Training: 60 backward graphs, 15 fixed updates, sole terminal sealed before four
target-blind CHECK forwards. Wall through finalization 141.755589042 s; peak
host RSS 3305816064 B, MPS driver 3250470912 B. Native zero3: 92.306243292 s,
peak 6650150912 B. Native terminal5: 153.823205084 s overall, peak 7283294208 B;
each call remained under its separate 120 s native deadline. Score: 8.696444292 s,
peak 352092160 B. All stages had zero added swap; completed external observation
intervals were <1 s; owned child cleanup and final seals passed. These are
instrumented native/resource checks, **not production-wall qualification**.

Root independently rehashed the final score inventory: 95 source bindings,
809 input bindings, 4 truth bindings, and 1 score artifact, all exact.

## Principal seals

- Decision: d90cec782ef7631acc227b1bb857f92d71284920521341764c3d182b87368d80
- Frozen COMP estimator: 2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a
- Terminal Q file: 36da3419d6a8c98b294f9359c26de3bae783f6e1d558a85dffe2e226d068c271
- Zero estimator: 211d59968a8014f4ca176355d4d9f6fee7bfb8603191cd695b9bef429c8b1a70
- Terminal estimator: 7da49c1c67c99e4f002d2d52dcb5e973532bb9908ec183cf33dacee6b919bb58
- Native-zero result: 3b472b38944266324ebd34ac48f5554cb2e14302075afd87e62430a2c35f6903
- Native-zero watch result: 1343b35104cc34472ad3f5edb1819675a68596cc6afc9f8d3d6ec8612735751e
- Native-terminal result: 75e8a104b7edee368ae898cfcbda661d0e5026553a245bd1bc3f189c1875493f
- Native-terminal watch result: 06cb032132591f76fc5841a1d420b1b4b498ea21b09e902bd4e2e66a382dd79f
- Score result: 8aebee62bd0700fbff8394307e8ef6926ae13bd498462188a6e0019c599eb7cf
- Score completion: bfda513a9f01cd870c372b3dbc5a71dad02cf6dbb460c21cb302ff21b5a31dc4
- Score finalization: e9f791b9068d31c7c7c95063a462e5a68c727f8591cbd75d2e480f928fc7fba3
- Score watch result: a3c42270f96e3d9015b42cddb3ce88addae90faff1aa772cef50612a10aa01da
- Score watch completion: da1f63556efee41c8251203ee375a027fa55d728ffe6126f7abd9b96ebc31811
- Score watch finalization: 67ae3de329df659391199c5bcb8ef402c8cbdbcae81792b62a29bc9bfc98f78d

Prior qualification/train/build chronology remains in the two preceding dated
result notes; it is not overwritten. The official confirmed bank is unchanged:
#330390, adjusted 4.53886349950395e-9, raw 2.120838555441651e-8. The <1e-9 goal
and <4/<3/<2e-9 validated milestone submission policy remain active.
