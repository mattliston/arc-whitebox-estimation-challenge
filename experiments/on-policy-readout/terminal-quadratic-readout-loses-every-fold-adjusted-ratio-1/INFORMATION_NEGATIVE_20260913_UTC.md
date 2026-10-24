# Fixed terminal quadratic information test: scientific negative

Actual run2026-09-13 UTC, session12039, fully retired. Operational worker and
external watcher PASS; fixed scientific gate FAIL. No full20 payload, candidate,
independent panel, upload or official improvement. Bank330390 unchanged.

Exploratory TRAIN-network-CV,20 networks, five fixed16-fit/4-held folds:

| Readout | Mean ordinary final MSE | Mean per-network adjusted product |
| --- | ---: | ---: |
| Unchanged current c2 | 2.1227343968756524e-8 | 3.894276334058273e-9 |
| Fixed affine | 2.1256422938638175e-8 | 3.89977093960885e-9 |
| Fixed quadratic | 2.288851073588472e-8 | 4.199756102846821e-9 |

Quadratic/baseline adjusted ratio1.0784432697076234. Joint fold wins0/5;
required4/5. Neither extra readout supplied a gain. The base head saw these
TRAIN networks; this is not untouched-network generalization. Original1B
FP32 labels, ordinary error (not noise corrected). Each network uses its
own measured cold model F plus fixed planning increments affine1MF and
quadratic10MF. Those increments are not a measured deployment bill.

All OOF predictions were sealed before aggregate scoring. Same frozen math,
normalization, ridge1e-4, one Cholesky/model/fold, actual denseFP32 held circuit.
No tuning, fold substitution, alternative solver or post-result association.
This closes the fixed quadratic H15 proposal; do not rescue it with a new
ridge, normalizer, seed, fit roster, rank or frozen-head variant. The affine
comparison is preserved too, not claimed as a successful small component.

Research work1545400778514F, including all five309078869120F folds and the
6432914F score context. Six contexts actually retired. Worker23.908404s,
watch33.090858s; peak tree1757134848B, zero swap growth, maximum completed
observation0.378811s. All numerical owners, processes and observation groups
retired, original captures/sources/labels unchanged.

Worker result/completion/finalization:
e51834845bb8460e816b7cecd0bc7b42b330558e355fd4ce4e4ba0dd0060758a
f42b4067b0887bd8d416af77173bf5c5fd439e0ef2aa0ab2d1733cd7380ee9c7
f983a123f9f92c9d42faa8174d4dbead1b54ef396a933251c3a02a20a265944a

Watch result/completion/finalization:
7d767b2ee00ddad8dce928ebc8dba7f0a74e2789c2842c7eede540845f235aac
a78345452ba57ec1f1d7ed6696171563ee905cb866cd9a2cbf4c5d9754cb31cc
3788b4f5b7ca6c00b388ad1508de2b8b1190ba4fb6b184bfd4fe039449633675

Next separate decision: finish actual preflight of the independently reviewed
whole-root batching unit and measure that fixed runtime component. This
negative raw test supplies no whole-card timing or new bank evidence.
