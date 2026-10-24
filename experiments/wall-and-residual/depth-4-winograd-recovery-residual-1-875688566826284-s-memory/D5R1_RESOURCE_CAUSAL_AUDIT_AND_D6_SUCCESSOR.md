# P2R0C1D5R1 resource-causal audit and sole P2R0C1D6 successor

Date: 2026-08-26

Status: **D5R1 CLOSED_THIS_IMPLEMENTATION_FORM_NO_RETRY. D6 OPEN,
DESIGN_ONLY, UNIMPLEMENTED, UNMEASURED, AND NOT AUTHORIZED TO RUN.**

This is a target-free, read-only terminal audit of the one D5R1 candidate call
and a preregistration-ready discriminator for exactly one materially different
successor. It does not open targets, scores, names, the board, or production;
does not train, submit, or call an estimator; and does not edit the canonical
graph. All score consequences below are explicitly projections.

## Immutable evidence bindings

| Artifact | SHA-256 |
|---|---|
| `public_noeigh_card_v1/estimator.py` | `373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7` |
| `estimator.py` (D5) | `69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5` |
| `RECOVERED_REFERENCE_EVIDENCE.json` | `cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6` |
| `RECOVERY_DESCENDANT_RESULT.json` | `a8221d116c03ed442cb6343a54b429b06e5a620f905c056fa99abe6497bc9549` |
| `RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json` | `8ee4d2be92f780b4f547e2129e59241cb19936a9767df1f04b3e9b6ba9b281c0` |
| `INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL_AUDIT.md` | `c17c38cffe77dd3f48de6bd0cf2554d92d5659c0a0c9bc4f9e94225b7b056de9` |
| `verify_independent_recovery_descendant_terminal.py` | `ede98b6732fe0e7b5eaeec3c6471da997ab2c4c14383a9fb8785c702f73e985c` |

The independent terminal verifier prints
`PASS_INDEPENDENT_REDTEAM_RECOVERY_DESCENDANT_TERMINAL`.

## What D5R1 established

Only ordinal 0, role-rank 35, ran. The exact candidate saved
`371,608,248,320` literal scalar operations against the exact incumbent and
kept very small numerical drift, but it failed both deployability gates and
stopped correctly:

| Measure | Incumbent reference | D5R1 | D5R1 change |
|---|---:|---:|---:|
| literal FLOPs | 1,692,484,904,311 | 1,320,876,655,991 | -371,608,248,320 |
| exact outer wall | 9.890440291725099 s | 24.356781082693487 s | +14.466340790968388 s |
| FlopScope backend | 8.859719954431057 s | 21.66671060444787 s | +12.806990650016814 s |
| FlopScope overhead | 0.648201156873256 s | 0.8143845787271857 s | +0.16618342185392976 s |
| residual wall | 0.3825201806612313 s | 1.875688566826284 s | +1.4931683861650527 s |
| peak process RSS | 2,950,840,320 B | 11,953,569,792 B | +9,002,729,472 B |

D5R1 was `2.4626589276386155x` the reference outer wall,
`2.4455299621080675x` its backend time, `4.903502250741216x` its residual,
and `4.050903639543599x` its RSS. Its residual exceeded the strict 0.4-second
gate by `1.475688566826284` seconds. RSS exceeded 8 GiB by `3,363,635,200`
bytes. The output was finite float32 `(16,1024)` with final-layer RMS drift
`4.109649235540117e-8` and maximum absolute drift
`2.384185791015625e-7`.

This was not a full `[35,279,35]` pass. Rank 279 and the repeat ordinal never
ran; D5R1's ordinal-0 repeat flag is vacuous; pooled drift and actual repeat
parity remain untested. No raw MSE was measured.

## Causal resource decomposition

D5 genuinely moved arithmetic from dense `einsum` into Strassen--Winograd,
but the fully nested `stack`/`block` allocation tree made inexpensive scalar
arithmetic and data movement dominate elapsed time:

| operation | calls, ref -> D5 | FLOPs, ref -> D5 | backend delta |
|---|---:|---:|---:|
| `subtract` | 495 -> 1,679 | 639,848,448 -> 8,804,896,768 | +4.6001374325715005 s |
| `stack` | 150 -> 456 | 270,661,632 -> 14,436,964,352 | +4.120183874852955 s |
| `add` | 1,024 -> 2,060 | 1,692,827,872 -> 7,104,954,592 | +1.8393290550448 s |
| `block` | 0 -> 128 | 0 -> 2,887,843,840 | +1.2078298297710717 s |
| `matmul` | 345 -> 352 | 1,016,019,814,400 -> 1,064,706,721,792 | +1.3954396164044738 s |
| `concatenate` | 92 -> 132 | 48,433,152 -> 300,091,392 | +0.03144374815747142 s |
| `einsum` | 174 -> 146 | 663,005,442,048 -> 212,063,236,096 | -0.31493953010067344 s |

The D5 `subtract + stack + add + block` group alone used
`33,234,659,552` FLOPs and `12.20690989587456` backend seconds, versus
`2,603,337,952` FLOPs and `0.4394297036342323` seconds in the incumbent: an
increment of `30,631,321,600` FLOPs but `11.767480192240328` backend seconds.
Its own four-operation-group FlopScope overhead was only
`0.2868644976988435` seconds. That isolates bandwidth-heavy materialization
and short-kernel execution, rather than the eliminated arithmetic, as the
wall-time cause.

The corrected D5 static peak was `8,037,105,664` bytes, but literal process
RSS was `11,953,569,792` bytes: a `3,916,464,128`-byte miss. The candidate
allocated paired recursive trees plus two persistent `(2,1024,16384)` read
workspaces; those two workspaces alone are `134,217,728` bytes beyond the
incumbent's product storage. The much larger projection miss shows that live
NumPy/BLAS temporaries and allocator high-water behavior were not bounded by
the static array ledger. That old ledger cannot authorize a nearby schedule.

Terminal causal verdict: the depth-4 arithmetic reduction survives as a
mathematical lead; D5's fully nested, allocation-tree and rank-gated paired
implementation form does not.

## Sole successor: P2R0C1D6

Label: **sequential-channel, fully preallocated depth-4 Winograd arena**.

D6 must preserve the incumbent/D5 CP3 contraction roster exactly: 30
channel-level push contractions (two channels at ranks 1,024 through 15,360)
and 28 channel-level read contractions (two channels at ranks 1,024 through
14,336), for 58 channel-level contractions total. The final K3 read remains
excluded. It changes only their matrix-product spelling:

1. Never batch the two CP3 channels. Reuse the incumbent's two full-cap
   product planes sequentially.
2. Replace the recursive allocator with four iterative Winograd transforms.
   At each level, build the complete bounded batch in two alternating,
   preallocated operand banks. The one leaf batch is `7^4 = 2,401`, exactly
   the same arithmetic batching depth as D5 but without recursive
   `fnp.stack` or `fnp.block` allocation trees.
3. At each transform level, compute the four `S` and four `T` dependency
   arrays once in vectorized reusable scratch, fill the destination bank with
   explicit `out=`, then swap banks. Reconstruction likewise alternates two
   fixed product banks plus one fixed `U` scratch region.
4. Every large add, subtract, copy, leaf matmul, and reconstruction writes
   through a supported explicit `out=` into a fixed float32 arena slice.
   Reconstruct directly into destination quadrants in the frozen Winograd
   order. No output-neutral dummy operation is allowed.
5. The schedule has exactly 58 channel-level depth-4 kernel invocations and
   therefore 58 leaf-batch matmul invocations per full prediction. This is 21
   more leaf matmul dispatches than D5's 37-call (20 push plus 17 read)
   rank-gated paired spelling but 348
   fewer than a one-level-streamed depth-3 spelling and 2,784 fewer than a
   two-level-streamed depth-2 spelling. The zero-stream spelling is selected
   because the worst recovered incumbent residual was already
   `0.3858333397656679` seconds: only `0.0141666602343321` seconds remain below
   the strict 0.4-second gate.

This is a design, not an implementation claim. Support for every proposed
`out=` shape must be proven by a target-free serialization/kernel smoke after
the source and accounting manifest are frozen. Unsupported overlap or a
hidden allocation is a static failure, not permission to retune the design.

### Honest memory admission

Let `q = 7/4`, the element-growth factor for a full Winograd batch at each
level. For one channel with `A=m*k`, `B=k*n`, and `C=m*n` elements, D6 adopts
the already independently checked conservative D5 recursion capacity rather
than a tighter new liveness estimate:

`sum(j=1..4) q^j*(A+B) + q^3*max(A,B) + q^4*C + 3*q^3*C + 2*C`.

This retains every recursive operand level, the deepest dependency transient,
leaf result, conservative recombination allowance, and two output banks at
once. The D6 manifest must round and bind exact, nonoverlapping slice intervals
for every term in one flat arena; it may not claim in-place reuse to lower this
admission bound.

At the worst push (`m=k=1024`, `n=15360`) the exact cap is
`844,165,120` float32 elements or `3,376,660,480` bytes. Added to the
conservative incumbent anchor `3,421,503,488`, the static application-array
total is `6,798,163,968` bytes, leaving `1,791,770,624` bytes below 8 GiB. The
largest read (`m=n=1024`, `k=14336`) is smaller: `681,480,192` elements,
`2,725,920,768` arena bytes, and `6,147,424,256` bytes including the anchor.
D6 allocates no additional persistent read planes; it reuses the incumbent's
two product planes sequentially. This is why D6 uses `6,798,163,968`, not the
corrected D5 sequential-push total `6,932,381,696`, which included two extra
full-cap read workspaces.

D5's static model nevertheless missed literal RSS by `3,916,464,128` bytes.
That miss is not silently transferred or ignored: D6's frozen static proof
must show that the dynamic allocation tree responsible for the old model is
absent, and a fresh target-free exact-shape kernel smoke must measure allocator
and BLAS workspace before any common-driver call. Its process-RSS increment
over its pre-arena baseline must be no greater than the exact arena cap plus
512 MiB, i.e. `3,913,531,392` bytes. Its literal absolute peak process RSS must
also be `< 7 GiB`; the anchor-plus-increment inequality is a separate static
cross-check, not a substitute for that absolute measurement. This preserves
at least 1 GiB for the later literal 8-GiB gate. The same smoke must compare
the complete 58-contraction roster
against the direct incumbent spelling and require D6's residual-wall delta to
be `<= 0.010` seconds. This leaves about 4.17 ms of the observed incumbent's
strict residual margin as contingency.

Before any common-driver call, an AST plus shape/liveness proof must bind every
large allocation to the formula above and prove that no recursive `stack`,
`block`, implicit full-size expression temporary, paired-channel tree, extra
persistent read plane, or second unbound product bank exists. Failure of the
static proof or exact-shape smoke closes D6 before execution. Literal per-call
RSS `< 8 GiB` and residual `< 0.4` seconds remain decisive.

### Frozen target-free discriminator

No common-driver execution is authorized by this document. If source,
builder, accounting manifest, static verifier, and independent audit are later
sealed, the one-shot target-free gate is exactly `[35,279,35]`, with one
persistent estimator in each fresh reference/candidate process and no target,
score, name, board, production, or submission access.

Before that gate, a sealed target-free literal kernel/accounting smoke must
measure, not project, the exact candidate bills. For role-ranks `[35,279,35]`:

- candidate FLOPs must be no greater than D5's exact trio
  `[1,320,876,655,991, 1,359,519,832,584, 1,320,873,503,232]`;
- measured incumbent minus candidate must be at least
  `371,608,248,320` on every call;
- literal operation sum must equal the total and total must be `< 2^41`;
- there may be no dummy billed work used to meet an expected number.

Every target-free common-driver call must also satisfy:

- finite raw float32 output of shape `(16,1024)`;
- exact outer and meter wall each `< 120` seconds;
- residual wall strictly `< 0.4` seconds and deployment residual `<= 0.385`;
- maximum process RSS strictly `< 8 GiB`;
- repeated rank-35 output byte-identical;
- all-roster and unique-fixture pooled final RMS drift each
  `<= 1.714909904361275e-6`;
- aggregate maximum absolute drift `<= 8.574549521806375e-6`, with every
  per-layer vector and worst ordinal/role-rank reported.

Any static, smoke, or common-driver failure closes this D6 implementation form
without retry. D5 candidate outputs cannot fill D6's unmeasured rank-279 or
repeat rows. The recovered incumbent evidence may be reused only through a
newly sealed, independently audited zero-reference-call descendant.

## Dedupe and closure boundary

The following exact prior sources were inspected:

| Family | SHA-256 | Why D6 is not a retry |
|---|---|---|
| Fable clean S3 | `c3c6981b4687724e8bbabf55c8103586b589bfbf0b2755bf3e71a584fef64ec9` | whole sample-forward product; recursive `stack`/`block`; depth 3 |
| Fable spend S3 | `4dee5200cb0c1c72eb5e9c9d8c0cecc989423363fdcd08b232e0e6bf4ead7820` | same full-batch allocation mechanism with a different spend policy |
| Fable spend7 S3 | `f6dc80663b9e35743e48141796e21087861bd1ea2e83015c3ebab38b9616b6d6` | same full-batch allocation mechanism with spend-7 policy |
| prior Winograd S3 | `87d790b1a24e08cfd6f18b38c8ffd7a31d21209ded4ba221cbeda18a7d9ec531` | full sample-forward recursion with `stack`/`block` |
| `K_probe_strassen.py` | `e68e478cdddc50f7d3a6cdd883a4bcbcb39faabbaaba02c323dd03dda1f5f65b` | compares stack/concatenate/setitem and block alternatives; no four-level CP3 alternating arena |

Fable's three S3 candidates recursively batch the entire sample-forward
product using `stack` and `block`; the prior Winograd S3 does the same with the
15-add law. D5 applies a recursive depth-4 allocation tree to CP3 and sometimes
batches channels. D6 keeps the depth-4 algebra but replaces recursion and
dynamic assembly with an explicitly liveness-bounded, iterative arena, writes
reconstruction directly, and never batches channels. `K_probe_strassen.py`
measured local assembly alternatives, including setitem, but did not implement
this full four-level alternating arena or apply it to the exact CP3 roster.

Closed scope: D5/D5R1's exact fully nested/paired implementation and any retry,
threshold-only retune, fixture switch, or reuse of its failed static margin.
Not closed: depth-4 arithmetic as a whole, the design-only D6 iterative-arena
form, or the broader accuracy-compute parent `P2R0C1D`.

## Score and graph disposition

At D5's measured call-0 bill, unchanged #328439 raw accuracy gives only the
planning projection `1.7876252782987673e-8` adjusted. Crossing adjusted
`1e-8` at that cost still requires raw below
`1.6451523794042476e-8`, about 44.1% below #328439. D5 measured no targets,
and D6 has neither an implementation nor a measurement, so neither is evidence
of that raw improvement or of adjusted `<1e-8`.

Graph-ready disposition:

- `P2R0C1D5R1`: `closed`, terminal `CLOSE_THIS_IMPLEMENTATION_FORM_NO_RETRY`;
- `P2R0C1D6`: `open`, design-only/unimplemented/unmeasured and implementation
  not yet authorized; terminal criterion
  is the complete static/liveness, literal-accounting, and independently
  sealed `[35,279,35]` discriminator above;
- `P2R0C1D`: remains `open`, because the broader accuracy-compute hybrid and
  D6's deployability are unresolved;
- `active_work`: empty. An open design is not an active run.
