# Absolute/squared graph-message q-head: terminal OOF receipt

Status: **FAIL_MATERIAL_OOF__STOP_NO_REFIT**.

This was the sole preregistered new target read.  The candidate concatenated
fold-pure local22 features, the previously tested signed raw20 recurrence, and
new target-blind normalized `abs(W)` and `W^2` raw20 messages (82 inputs).  A
shared 256-by-3 SiLU trunk with six layer heads was trained for 13 fixed epochs
using the inherited deterministic 256-node sampler.  Held scoring used all
1024 nodes.  No feature ablation, hyperparameter search, observer-selection
target, estimator call, board access, or all-688 refit occurred.

## OOF result

| Metric | Value |
|---|---:|
| Candidate endpoint raw MSE | `2.9728904042473882e-8` |
| Saved fold-pure shared256 control | `2.985249910997227e-8` |
| Absolute gain | `1.2359506749839335e-10` |
| Relative gain | `0.4140191648380495%` |
| Preregistered 1% threshold | `2.9553974118872554e-8` |
| Current-board raw reference | `2.9409071836e-8` |
| Candidate minus current board | `3.198322064738811e-10` |
| Exact q-oracle endpoint MSE | `6.141640911984781e-9` |
| Candidate local delta-q MSE | `5.254981674768097e-9` |
| Network win fraction vs control | `0.6162790697674418` |
| Paired network z vs control | `7.78298417926875` |

Every fold improved, but none of that licenses continuation because the
aggregate missed the frozen materiality gate:

| Fold | Candidate | Saved shared256 |
|---:|---:|---:|
| 0 | `3.0023717906766396e-8` | `3.01886482724548e-8` |
| 1 | `2.9492970899820393e-8` | `2.9551042589062662e-8` |
| 2 | `2.968070596804098e-8` | `2.9771224993200252e-8` |
| 3 | `2.9718221395267752e-8` | `2.989908058517138e-8` |

The `W^2` channel has partial prior art in a one-endpoint mechanism screen.
`abs(W)` raw20 messaging is the binding novelty, but this single combined run
does not identify its isolated contribution and no post-outcome ablation is
authorized.

## Cost and legality

- Target-blind message construction: `0.48828125` width-GEMM equivalents per
  network (`13` signed, `6` absolute, and `6` squared transports of 20 fields).
- Head forward pass: `0.87158203125` width-GEMM equivalents per network.
- Combined message plus head forward estimate: `1.35986328125` equivalents.
- Head parameters: `154,374`.
- Strict deployment remains conditional on the native raw20 and matching
  source-row/target-column weight buffers being live at the same suffix cuts.

## Frozen execution receipt

- Exact command:
  `nice -n 10 env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /Users/matt/anaconda3/bin/python3 pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_abs_sq_message_qhead_oof_v1/run_oof.py`
- PID: `29412`.
- Start: `2026-08-31T16:14:17.409461+00:00`.
- Completion receipt written by `2026-08-31T16:22:33Z`.
- Program SHA-256: `e5028cfe8cd088abc6f4997d9995f2c6b1ba65afd1ee280e763856578a6390ba`.
- Preregistration SHA-256: `3e30b7f9c9ada81a7e15b2e6ac3792af5f47ddab7c176f808b162fafa665a4d1`.
- One-use claim SHA-256: `64bd82fd4961c63c8fe413ca842e450f4b35b703d7992886b80687d381b83a68`.
- Result SHA-256: `87624c36e90e3db7e5592ff8eb56acf5696a37d9ad0f2dd7ed656394301f98b4`.
- OOF model bundle SHA-256: `21572ccd20eb339057c9c32a127fd794c14355dc654df517a27091000ee807d2`.
- Absolute/squared cache summary SHA-256:
  `180e2f3458f64552583877dec80c8558e61bf0bebda7676f0921df9db823e9e9`.
- Signed cache summary SHA-256:
  `fbe6656b064847ee5895f87e4333ff4753bb5339b0bd78d8353371d6bbf0aa9d`.

All 1,376 live cache-row files were rehashed against their row receipts before
the durable no-retry claim and first semantic target decode.  The target file
was byte-hashed preclaim but not semantically decoded until afterward.

Decision: **close this spelling; do not refit or train outcome-selected
abs-only/squared-only variants.**
