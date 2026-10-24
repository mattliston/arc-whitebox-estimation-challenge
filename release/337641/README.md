# Submission #337641 — release materials

Public leaderboard score: **1.4006490071107744e-9** (100 of 100 networks graded, zero failures). Fired 2026-10-21T22:29:33Z as release tag `fable_hgfh_s135_c4s8e9_clean_govstrip3_5a4c1540_v1`.
Designated for the Phase 2 Private Re-evaluation together with #337619.

## What is in this directory

| path | what it is |
|---|---|
| `submission.tar.gz` | the exact artifact uploaded to AIcrowd (sha256 prefix `e0a79ccb4c52`), byte-identical copy |
| `extracted/estimator.py` | the single source file inside the artifact (sha256 `5a4c15405c446270…`); every precomputed table it needs is embedded in the file (base64 + zlib), it imports nothing but `flopscope`, `flopscope.numpy`, `whestbench.base` and the stdlib modules `base64`, `gc`, `math`, `time`, `zlib` |
| `extracted/manifest.json` | the packager manifest inside the artifact (entrypoint `estimator.Estimator`, estimator sha256) |
| `packaging/` | the `whest validate`, `whest package` and `whest validate-package` records (whestbench 0.14.0) and the stage-A gate record (einsum-signature audit, Python 3.10 compile check) |
| `gates/` | the local release gates run on this exact file before firing: 5-worker setup timing (`setup5/`), prefire import audit, gate log |
| `hosted_grade/` | the hosted grading record scanned from the public submission page (`SUMMARY.md` and the raw JSON) |
| `build/BUILD_v1.json` | how this file was built: parent seal + appended tail (recorded verbatim in the JSON) |
| `build/candidate_*_science_v1.py` | the local-evaluation twin of the released file (same estimator; a science-mode wrapper used for paired local scoring) |
| `build/tail_sm40_cap4_size8e9_xsg_off.py` | the appended tail: Smirnov-plan size drop (products >= 8.0e9), Smirnov read cap of 4 per predict, scatter-for-gather off |
| `seal_s1.35/` | the parent of the build: the on-policy readout seal at strength x1.35 (`EXPORTED_v1.json` records parents, coefficients and the step override) |

The retired residual-governor helper functions (never called: their dispatch table `_FAST_RESCUE_FRAC` is empty) were replaced by `return None` before packaging; `build/` holds the pre-edit copies (`*_before_govstrip.py`) so the edit can be diffed.

## Lineage (each step is a file in this package)

1. **Parent estimator** = submission #337506 lineage (`common/parents/`): release file sha256 `f9381d09…`, local science twin `49bd2ba3…`.
2. **On-policy second-step readout** (`common/readout/`): the parent file was run on 90 training networks drawn with fresh seeds from the public starter-kit generator; per-layer features of its own trajectory were captured (`first128_onpolicy_hook_*.py`, `export_first128_onpolicy_tensors_v1.py`, capture scripts) and a 961-column ridge regression (ridge 0.03, damping 0.5; gate design 705 columns + hidden x gate 128 + first-128 x hidden 128) was fit per layer (`fit_blocks_full90_v1.py` with the `lc_*` design modules) -> `full90_BLOCKS_COEFFICIENTS_v1.json`.
3. **Seal export** (`export_blocks_v1.py`): the coefficients are written into the parent as an override of `_gop2_gnn_step`, scaled by strength x1.35 -> `seal_s1.35/candidate_first128_onpolicy_blocks_release_v1.py` (sha256 `a277c387a40b0ed9…`).
4. **Stack build** (`build/BUILD_v1.json`, 2026-10-21T21:15:56Z): seal + tail -> `candidate_*_release_v1.py` -> governor-strip edit = `extracted/estimator.py` (sha256 `5a4c15405c446270…`).
5. **Package and gates**: `whest package` -> `submission.tar.gz` (`e0a79ccb4c52`); setup5, stage A and prefire gates in `gates/` and `packaging/`.

## Re-running

The artifact follows the WhestBench estimator contract (class `Estimator`, `setup(context)` once per worker, `predict` per network, all arithmetic through the flopscope client). The packaging records show the exact commands used:
`whest validate --estimator estimator.py --class Estimator`, `whest package --estimator estimator.py --class Estimator --output submission.tar.gz`, `whest validate-package submission.tar.gz`.
Dependencies: see `../DEPENDENCY_MANIFEST.md`. License: see `../LICENSE`.
