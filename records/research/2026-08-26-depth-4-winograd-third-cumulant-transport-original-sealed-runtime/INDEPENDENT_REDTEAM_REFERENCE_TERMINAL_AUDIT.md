# Independent red-team audit: terminal reference harness failure

Date: 2026-08-26

Verdict: **the three reference science calls all passed; the original sealed
protocol nevertheless failed terminally and is closed.** The failure is confined
to final receipt construction after the three-call loop. No candidate call ran.

## Evidence verified

The exact combined PTY stream is `REFERENCE_EXECUTION_TRANSCRIPT.txt`, SHA-256
`5386d71702e08be689bd0a48bea12eeee47264354f263ce6df9976ce2222fd28`.
The terminal receipt is `REFERENCE_TERMINAL_FAILURE_RECEIPT.json`, SHA-256
`4bf9f7c6c43621cfa167860ad5d10f08628c858e48000e00edc45e4d74c25157`.
Both bind the original 38-file seal
`776859ea1d510d7e7178533c888100fad23ac82d7412a03e1898bf77b4f703f0`.
The preserved session log independently establishes the original command, its
three streamed trial records, traceback, and exit code 1.

The exact `[35, 279, 35]` reference measurements were:

| Ordinal | FLOPs | Outer wall (s) | Meter wall (s) | Residual (s) | Peak RSS (B) |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,692,484,904,311 | 9.890440291725099 | 9.890441291965544 | 0.3825201806612313 | 2,950,840,320 |
| 1 | 1,731,128,080,904 | 10.023627415765077 | 10.023652957752347 | 0.3858333397656679 | 3,305,242,624 |
| 2 | 1,692,481,751,552 | 9.825676208361983 | 9.825698040891439 | 0.37958328472450376 | 3,334,782,976 |

For every ordinal, the printed literal operation sum equals the literal FLOP
total and every recorded reference gate is true: exact bill, operation sum,
sub-2^41, outer and meter wall, residual below 0.4 seconds, RSS below 8 GiB,
raw float32 `(16, 1024)` schema, finite output, and repeat parity. The three
output files match their printed file and array hashes. The two rank-35 files
are byte-identical.

The rank-279 reference residual of 0.3858333397656679 seconds passes the
reference's strict 0.4-second gate but is 0.0008333397656679 seconds above the
candidate deployment threshold of 0.385 seconds. That does not fail reference
capture, but it confirms the candidate residual-wall gate has little margin and
must remain empirical and terminal.

## Exact failure scope

After ordinal 2 printed a conjunctive pass, control reached
`finish("PASS_REFERENCE_CAPTURE", seal, trials)` at line 222. Receipt
construction then evaluated lowercase JSON spelling `false` at line 138, which
is an undefined Python name. The resulting `NameError` occurred before
`REFERENCE_RESULT.json` was written. The never-run candidate runner contains
the same latent receipt-only defect at lines 193--194.

This is not a numerical, FLOP, resource, fixture, source, or repeatability
failure. It is still a terminal harness failure under the frozen protocol.
Therefore:

- original protocol status: `FAIL_TERMINAL_NO_RETRY`;
- `REFERENCE_RESULT.json`: absent;
- candidate outputs/result: absent;
- candidate launch count: zero;
- patching or rerunning the original sealed runner: rejected.

## Zero-reference-call descendant disposition

**AUTHORIZE design, static audit, and sealing of a fresh recovery descendant;
do not yet authorize its candidate execution.** A descendant can preserve the
scientific no-retry rule because it needs zero additional reference estimator
calls and the candidate has never run. It must not claim that the original
protocol passed or resume/modify the old seal.

Launch authorization for such a descendant requires all of the following:

1. Preserve the original seal, transcript, failure receipt, output files, and
   failure status byte-for-byte.
2. Generate a distinctly named recovered *scientific-evidence* receipt solely
   by parsing the three transcript JSON records and hashing/validating the three
   existing arrays. The recovery tool must not import an estimator, construct an
   `Estimator`, call `predict`, access targets, or perform training.
3. Bind the session evidence, original exit code 1, call order, exact bills,
   operation dictionaries/sums, timings, RSS, gates, output hashes, and
   traceback. Never relabel the original process as a successful receipt run.
4. Use a new candidate-runner filename, result filenames, manifest, and seal.
   Do not alter either original runner. The new candidate runner may fix only
   harness/receipt plumbing; candidate estimator SHA-256 must remain
   `69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5`.
5. Add an AST guard rejecting loaded names `false`, `true`, and `null`, plus a
   target-free unit test of receipt serialization that performs zero estimator
   calls. A parse/compile check alone is insufficient because lowercase `false`
   is syntactically valid Python.
6. Execute exactly one fresh candidate process with one persistent estimator and
   calls `[35, 279, 35]`, no retry or retune. Gate measured recovered-reference
   FLOPs minus measured candidate FLOPs at exactly 371,608,248,320 per ordinal,
   together with the already frozen absolute bill, drift, schema, finite,
   repeat, wall, residual, and RSS gates.
7. Stop terminally on any recovery, seal, or candidate failure. No targets,
   scores, names, board access, production, submission, or network access are
   authorized.

This disposition authorizes preparation only. A separate independent review of
the exact recovered receipt, descendant runner, manifest, verifiers, and seal is
required before the first candidate call.
