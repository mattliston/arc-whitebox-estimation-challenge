# Dependency manifest (both submissions)

The released estimator is a single Python source file with no supporting modules of its own. Its imports, verified on the archived files:

| import | role |
|---|---|
| `flopscope`, `flopscope.numpy` | the competition's metered array client; every arithmetic operation of the estimator goes through it |
| `whestbench.base` | the competition's estimator base class (`BaseEstimator`) |
| `base64`, `zlib` | stdlib; decode the precomputed tables embedded in the source file |
| `math`, `time` | stdlib; scalar bookkeeping and timing |
| `gc` | stdlib; freezes the setup object graph and raises the gen-2 collector threshold during `predict`, restoring the previous settings afterwards (keeps collector pauses out of the residual wall; performs no computation) |

There is no NumPy import, no compiled extension, no thread, process, FFI or network use. All precomputed artifacts (moment tables, learned per-layer corrections, the on-policy readout coefficients) are embedded in `estimator.py`; no separate data file is needed.

## Versions recorded at packaging time (`extracted/manifest.json`)

| component | version |
|---|---|
| whestbench (packager and validator) | 0.14.0 |
| flopscope client (local packaging environment) | 0.10.0 |
| Python (local) | 3.11.5; manifest `min_version` 3.11; stage-A gate also verified the file compiles under 3.10 |
| numpy (present in the local environment only; not imported by the estimator) | 2.2.6 |

At evaluation time the file runs in the grader's own pinned environment (flopscope client, WhestBench participant interface and the Python standard library), as specified in the Official Rules §5.6. The hosted graders that produced the leaderboard scores recorded flopscope-metered FLOPs of 27687291204 per setup execution and mean compute utilization 0.108659 (#337641) / 0.108652 (#337619).
