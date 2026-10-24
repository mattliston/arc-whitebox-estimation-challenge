# Fixed independent covariance/mean endpoint: efficacy failed

Recorded September 9 Pacific / September 10 UTC from completed receipts.
Phase 2 only. This is not a submission or an independent-panel result.
Official bank #330390 remains adjusted 4.53886349950395e-9.

## Actual objective

The sole terminal endpoint in DECISION.md completed qualification, its
fixed fit, literal builds, native zero admission, native CHECK4 acquisition,
and sealed scoring. Every integrity/resource gate passed. Both prospective
scientific advancement gates failed. No fit extension, sweep, native40,
independent14, production test, or submission is advanced by this result.

CHECK ranks 37 (cold), 39, 42, 48 (warm); repeated 37 is excluded. Adjusted
means below average each network's final raw MSE times its own complete
max(.1,F/2^41), not a product of panel averages. Archived controls were
reauthenticated and replayed exactly without rerunning or refitting them.

| Configuration | Final raw MSE | Mean complete FLOPs | Adjusted score |
|---|---:|---:|---:|
| Original COMP | 1.9933680058595472e-8 | 475126242316.25 | 4.2964338254612944e-9 |
| Matched P-only control | 1.986607659226598e-8 | 475126242316.25 | 4.281556597677963e-9 |
| Sole P+covariance terminal | 3.504779533236127e-8 | 475231339973.25 | 7.589299078726298e-9 |

Terminal/P-only raw ratio 1.764203171652205, adjusted ratio
1.772556056561821, complete-F ratio 1.0002211994363597, **0/4 wins**.
Terminal/COMP adjusted ratio 1.7664182405768718, also 0/4 wins.
Thus the endpoint failed on accuracy, not an unexpectedly expensive action.

The already-preserved P-only screen remains a small component observation:
adjusted ratio .9965373078260472 against COMP, same compute, 3/4 wins.
It is not newly validated on native40, independent14, production, or board.

## Scope and faithful compatibility

Start: original COMP candidate
`2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a`.
P11 was trainable; original A/cstar/non-P and 66 inactive coordinates were
byte-preserved. A(H(P)) and endogenous physical trajectories remained live.
The added Q was FP32 [14,65], initialized to positive zero. Its prescribed
congruence acted after current mean recentering/active diagonal restoration
and before Bpost, trace, innovation, newborn and star reconstruction.
It was a C-only compensator, not a claim to rescale the entire joint law.

Exactly 60 TRAIN backwards, 15 updates, one terminal seal, and four
target-blind CHECK forwards were executed. The seed, TRAIN20 order,
batching, P optimizer, loss and original target dtypes matched P-only.
Both normalized cost-table files had SHA256
`09c1fa66ac56fc9a67bfa136805c3e55a722ba5ebc56380c7da9fdd6d0bb73bf`.
Their experiment-specific paths differ; content/hash equality was fixed in
the scorer before its sole execution, not by weakening loss compatibility.

Online epoch-mean losses were 2.345296947758868e-7,
2.3756103839644993e-7, and 5.642898504021332e-8. These are moving-parameter
training observations, not fixed-endpoint scores or evidence that identifies
the cause. This failed optimizer/parameterization endpoint does not prove
that all covariance corrections are useless or establish a capacity ceiling.

The zero candidate passed all16-output/47-selector and own-repeat checks.
Its directly measured complete delta was 105099022F cold and 105097202F
warm; the 1820F difference is Q initialization. Actual cold startup was
1383916118F. Terminal bills were measured separately, never borrowed from
the zero trajectory. All CHECK predictions and bills were sealed before
opening the four cached truths. Root independently rehashed the scorer's
96 source, 1236 input, four truth and one artifact bindings afterward.

## Resource and retirement receipts

One serial nice-10 numerical worker, BLAS/Torch threads1, one MPS job,
fallback0/lowwatermark.1; no AP threadpoolctl. Observed swap remained
772538368B throughout. Every stage and owned process tree retired, with
external completed-sample intervals below one second.

| Stage | Retired full calls | Watch elapsed s | Maximum observed combined RSS bytes | Maximum sample interval s |
|---|---:|---:|---:|---:|
| Qualification | 5 | 23.630925458 | 1659322368 | .355496792 |
| Fixed training + CHECK forwards | 64 | 247.258133458 | 4269375488 | .378123292 |
| Zero build | 0 | 6.864233834 | 352354304 | .325268417 |
| Terminal build | 0 | 8.683585 | 347291648 | .332440625 |
| Native zero | 3 | 91.916457958 | 6740967424 | .332640166 |
| Native terminal | 5 | 152.590286291 | 7028686848 | .331850292 |
| Saved-output scoring | 0 | 16.968877792 | 405536768 | .324592167 |

Qualification and training driver maxima were 2185019392B and 3250585600B.
Separate native input admissions also passed and retired. Native diagnostic
timing is not production timing; no production qualification is claimed.

## Preserved identifiers

- Decision: `814ac1f31121db92f3165ecfa0cb3253e8b66617e3e62d17316f2f288e71be98`.
- Terminal P: `95e1ca05ab8278093fefd015672cf4ffd5410d351a4933ef2ec9e647e0278a5e`.
- Terminal Q file: `642d6528712a5880c38e5219df61a8d9be0b12a6b928867a0537f62b81a3b36b`.
- Zero participant: `224c3d02a30493cc035af895c830b927af5d8d39d2e9796e7a1672dd26cef9be`.
- Terminal participant: `fb01bd0c93d5e88b2740778cb4c78c0891f88d831595d052de4d12b640ab9229`.
- Native terminal result: `04e98d0847338c98f96174464a748b6cdfae5323d425666234a3537926df4dd8`.
- Score result: `46565caba59ad9b528c80587b1cd8bdf9ec30ef1232e83a98229cbf0ad710e5d`.
- Score completion: `a79e9d4a0bb2195dd6788ff094d7946b22552309b0b1eca03e975c7ab7ca073f`.
- Score finalization: `d794df9578f5643d2c872aa658df6e17979b666be744d9c149568b66b3b94ff6`.
- Score-watch result: `d85df606857c92002868d75c82f80da37c16671ac14a83ccf707e6f0574f8382`.
- Score-watch completion: `c8730cbdfdc349569f4a3fb3224c221d459224a742f008623cf4a83e6dbd5929`.
- Score-watch finalization: `3a7060379b0508dbc67aa313abbe33aacded3a6b0815208c5ddc29ee3f1ce7ca`.

Source-only failed watcher preflight is separately preserved in
`../watch_native_covariance_source_preflight_failed_v1.py`; it launched no
numerical worker. All experimental and older failed artifacts remain intact.
