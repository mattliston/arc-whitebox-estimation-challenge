# PREREGISTER — P2R0G19B_REFIT native-collective corrector refit

Written before any training in this experiment (the `folds/` directory is
empty at prereg time; the completed-fold artifacts bind to
`run_experiment.py` sha256 `693a6d1dc5d6872ddff8a017b23b0fea6598a4507d592b4e2a42e0f5ccbdbc84`,
which postdates this file).

## Node binding

This experiment binds to graph node **P2R0G19B_REFIT**. It does NOT bind to
P2R0F2Q — that node is closed; the `f2q` in the directory and feature-store
names is historical.

## Features (as implemented)

The three added local features are AS IMPLEMENTED by the exporter
(`public_collective_card_v2/export_coll_features.py`, sha256
`68a221977aff082d12462aa1ef6a17cd0830b57740419b3a45490574b89dd528`):

    col 0: K3S_l * w_i^2
    col 1: w_i
    col 2: (K3S_l / sigma_l) * b_i

The third column is the (K3S/sigma)*b delta-proxy form, and this
preregistration resolves the `delta_l*b_i` wording in the graph in favor of
the implemented formula above.

Feature corpus: `public_f2q_coll_features_v1/rows/{rank:04d}.npy`, 688 files,
each (16, 1024, 3) float32, verified finite and unique; corpus sha256 (sorted
`rank sha` lines) `3a59f12d231a6a2e437c0466b7bca5fcfcf61ce72d94dcd313353153113197ad`
(per-file hashes and column stats in `CORPUS_MANIFEST.json`; loader
`dataset.py` sha256 `babd7733966f7b94b358a5b30906bcf19c8cbe47eb49a8e9cb7e49f8f7f1e3c1`).

## Baseline and fold structure

Baseline = F2J (`public_full688_transport_energy_coherence_cv_v1`), OOF final
MSE **3.2334958229470984e-8**. Fold structure identical to baseline: same
SEED 1_753_358_438, same blake2s fold_of hash, 4 folds, sizes
[147, 172, 171, 198]. Architecture and every hyperparameter identical to the
baseline; only the input width changes (locals 20 -> 23, INPUT_FEATURES
36 -> 39, parameters 25_600 -> 25_888).

## Sentinel halve gate (binding, verbatim)

The sentinel "halve" gate means the new corrector's held-out MSE on that
exact fold must be <= 0.5x the F2J baseline's held-out MSE on the SAME fold
(matched fold, matched nets), with no retry and no threshold revision after
seeing numbers.

The sentinel fold is fold 0 (first in the baseline's fold order). The matched
same-fold baseline number IS recovered from banked F2J artifacts — no
recompute needed:
`public_full688_transport_energy_coherence_cv_v1/folds/fold_0.json`,
autonomous final MSE **3.154306195654431e-8** over its 147 held-out networks.

Decision table (fixed before training):
- reduction >= 50% vs the matched fold AND most nets in the fold improving
  -> proceed to the remaining folds, then full report;
- reduction 8-50% -> STOP after the sentinel, REFIT_RESULT.json status
  "sentinel-partial" with numbers and per-net deltas;
- reduction < 8% or negative -> STOP, status "sentinel-fail"; the lane closes.

No packaging, metering, or board-related work in any branch of this
experiment.
