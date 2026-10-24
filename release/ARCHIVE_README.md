# ARC White-Box Estimation Challenge 2026 — Phase 2 release package (Matt Liston)

Release materials for the two Phase 2 submissions designated for the Private Re-evaluation, assembled 2026-10-22T00:42:18Z from the archived build and release records (every file is a verified byte-identical copy; see `COPY_RECEIPT_v1.json` and `SHA256SUMS.txt`).

| submission | public leaderboard score | fired (UTC) | what it is |
|---|---|---|---|
| [#337641](337641/) | 1.4006490071107744e-09 | 2026-10-21T22:29:33Z | on-policy readout at strength x1.35 + Smirnov cap-4 / size-8e9 tail, scatter-for-gather off, retired governor code removed |
| [#337619](337619/) | 1.4005566073702658e-09 | 2026-10-21T20:39:06Z | the same estimator at readout strength x1.5 (retired governor code present as dead code) |

Both graded 100 of 100 networks with zero failures. The two files differ only in the readout strength constant and the removal of dead code; they are a dead tie on the public leaderboard.

## Layout

- `337641/`, `337619/` — one directory per submission, each self-contained for the open-source release requirement of the Official Rules §8: the exact uploaded artifact, the extracted `estimator.py` and manifest, the packaging and gate records, the hosted grading record, the build record and the parent seal. Each has its own `README.md`.
- `common/` — provenance shared by both: the parent estimator files (`parents/`), the on-policy readout capture/fit/export code and the fitted coefficients (`readout/`), the appended tails (`tails/`), the governor-strip tool (`tools/`), the technical write-up and compliance disclosure draft (`writeup/`), and the submission-ledger excerpt.
- `DEPENDENCY_MANIFEST.md` — imports and versions. `LICENSE` — MIT. `SHA256SUMS.txt` — checksums of every file in this package.

## How the estimator works (short)

A deterministic moment-closure estimator for the mean post-ReLU activation of each neuron of a random 1024 x 16 ReLU MLP under standard-normal input: layer by layer it propagates per-unit means and variances and a bank of learned, low-rank corrections (third- and fourth-cumulant "planes", Smirnov-sketched joins, per-layer dial tables), all computed through flopscope within the metered budget; a second-step linear readout, fitted on captures of the estimator's own trajectory on 90 training networks, corrects the late-layer means. The write-up draft in `common/writeup/` describes the method, the measurements behind each component, the use of LLM coding agents in its development, and the parts whose mechanism is understood versus empirical.

## Integrity

- Archive sha256 prefixes: #337641 `e0a79ccb4c52`, #337619 `55a93ef002c2` (match the release tags and the ledger).
- `extracted/estimator.py` in each directory is byte-identical to the `candidate_*_release_v1.py` that the build record names (sha256 `5a4c1540…` and `f66cb28b…`).
- No file of the research tree was moved or modified to make this package; `COPY_RECEIPT_v1.json` lists each copy with its sha256.
