# Independent D7 dual-launch failure and D7R1 eligibility audit

Date: 2026-08-26

Verdict: **CLOSE ORIGINAL P2R0C1D7 FOR PROTOCOL/RACE; PASS D7R1
PREPARATION ELIGIBILITY ONLY; NO RECOVERY RUNTIME, EXACT-SHAPE, OR ESTIMATOR
AUTHORIZATION.**

## What happened

Two agents independently created launch seals in the same path and launched
the same frozen runner before learning of the other launch.  This is a
one-shot coordination failure.  The original D7 authority is consumed and
must not be reused, even though neither process reached the scientific
discriminator.

The two immutable attempt chains are:

| attempt | artifact | SHA-256 |
|---|---|---|
| implementation-agent launch | `TINY_RUNTIME_LAUNCH_SEAL_ATTEMPT_001.json` | `8aeb6adc651a8292dea39be53f6d3c01f66c0a4a9e82be257e05bc9253fb31fa` |
| implementation-agent launch | `TINY_RUNTIME_LAUNCH_ATTEMPT_001_TRANSCRIPT.txt` | `0e244c00291c6a915f2129082df5bed2510f7ffae4df00c49a96086207281417` |
| implementation-agent launch | `TINY_RUNTIME_LAUNCH_ATTEMPT_001_FAILURE.json` | `5bdcc2530ff208843de0dc169bbc8d8b17746e75f12a1829b626531d24d0bbff` |
| tiny-authority launch | `TINY_RUNTIME_LAUNCH_SEAL.json` | `72c7bc9e264f63de2f6bcbe5129b927ed006bfe41b4cbbb98e7282c35c379602` |
| tiny-authority launch | `TINY_RUNTIME_LAUNCH_ATTEMPT_002_TRANSCRIPT.txt` | `4ac804fd6b724871362fb274216ca640d0e19ea1806b01c512d3707540b949f2` |
| tiny-authority launch | `TINY_RUNTIME_LAUNCH_ATTEMPT_002_FAILURE.json` | `6084d8f0e7f36be2c82ac5c67cf2ccb0b0dc3de719ee171adee6083587f33bcf` |

Both seals bind the same kernel source
`81d14f2ca983c2820ab078d15a3a8a861427ac5a8b1b5e82828300256648bbe9`
and runner
`cb4c8a6b0a51ab43adae83a6da0f2f77d522286a8815f4a28440f43ff899f048`.
Both transcripts show exit status 1 at runner line 198:

`assert bind_inventory["flops"] == 84`.

The implementation-agent command first attempted `nice -n 10`; the priority
change was denied, but Python proceeded and reached the same assertion.  The
tiny-authority launch called the data-gate Python directly.  There is no
`TINY_RUNTIME_RESULT.json`.

## Exact execution boundary

Each process verified its seal, imported only the isolated kernel module, and
called `bind_coefficients()`.  That helper made three `fnp.asarray` calls.
The assertion fired before scaled input creation and therefore before output
allocation, the gapped case, `staged_winograd`, direct `matmul`, or any timed
arm.  Across both launches:

- staged candidate calls: 0;
- direct baseline calls: 0;
- gapped semantic calls: 0;
- timed repetitions: 0;
- estimator, fixture, common-driver, exact-shape, target, score, name, board,
  production, submission, and network calls: 0.

The observed bind value was not printed or serialized by either frozen
runner.  This audit does not claim it as an observation.  The corrected value
below follows statically from the exact environment's billing law.

## Static cause and corrected cold bill

`U`, `V`, and `W` each contain 28 Python `float` values.  For
`fnp.asarray(tuple, dtype=float32)`, FlopScope 0.12 first probes the tuple as a
float64 NumPy array, then materializes a distinct float32 result.  Its
`asarray` law bills a materialization at the heavier source/destination dtype.
The official `asarray` weight is 1, the float32 rate is 1, and the float64 rate
is 2.  Therefore the exact static law is

`3 tensors * 28 elements * float64 rate 2 = 168 FLOPs`.

The supporting exact-environment source hashes are:

| source | SHA-256 |
|---|---|
| `flopscope/_array_ops.py` | `e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c` |
| `flopscope/_dtype_billing.py` | `d6f4df8b2fcf162f41afde2dae23a66d894eba62f173e57d89051e5873569cb8` |
| `flopscope/_budget.py` | `ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01` |
| `flopscope/data/default_weights.json` | `f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a` |

The corrected binding inventory is three `asarray` records, 168 total billed
FLOPs, and 168 `asarray` cost.  This changes no hot kernel or roster law.  It
adds 84 FLOPs only to the first projected call:

- corrected projected D7 calls: `[1,431,110,075,935,
  1,469,753,252,360, 1,431,106,923,008]`;
- corrected incumbent-minus-D7 savings: `[261,374,828,376,
  261,374,828,544, 261,374,828,544]`.

## D7R1 preparation eligibility

A distinct `P2R0C1D7R1` recovery descendant is scientifically eligible for
preparation because both failed launches stopped before any candidate or
baseline call and revealed no residual, drift, hot bill, or resource outcome.
This is not authorization to rerun original D7.

The recovery must satisfy all of the following before a new runtime review:

1. Preserve the kernel source, coefficient tensors, shapes, seeds, numerical
   gate, residual gate, repetitions, thread count, and hot accounting exactly.
2. Correct only the cold-binding law and dependent first-call projection to
   the values above, and use distinct D7R1 manifest, runner, result, seal,
   transcript, and attempt-marker paths.
3. Bind both failed attempt chains and this audit so the dual launch is not
   hidden or relabeled as one attempt.
4. Designate exactly one launch owner; every other agent must explicitly
   stand down.
5. Acquire an atomic `O_CREAT|O_EXCL` attempt marker before importing the
   kernel.  A second process that sees the marker must exit before import and
   must not consume another scientific attempt.
6. Persist the binding inventory before its assertion and leave the attempt
   marker in place on success, failure, or interruption.  Any exception after
   marker acquisition must produce or be paired with a terminal transcript.

If those conditions pass a new independent pre-run audit, one D7R1 tiny-only
execution may be considered.  This document authorizes preparation and
review only.  It authorizes no runtime call, exact shape, estimator, fixture,
common driver, target, score, name, board, production, submission, graph edit,
or network access.
