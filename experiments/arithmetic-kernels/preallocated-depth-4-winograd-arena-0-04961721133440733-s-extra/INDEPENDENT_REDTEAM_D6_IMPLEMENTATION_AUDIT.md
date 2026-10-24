# Independent red-team audit: P2R0C1D6 fixed-arena implementation

## Terminal disposition

**PASS static derivation, memory formula, billed-operation conservation, and
sealed tiny semantics.  FAIL the prerequisite residual-wall gate.  Close this
D6 implementation form.  Do not authorize the exact-shape smoke, estimator,
fixture, or common driver.**

The implementation is algebraically sound and its advertised compute saving
is real.  It is not launchable because a same-source, same-FlopScope, sealed
tiny 58-contraction diagnostic measured a median D6-minus-direct residual of
`0.04961721133440733 s`, versus the frozen maximum `0.010 s`.  All eight paired
deltas were between `0.04947213642299175 s` and `0.052486052736639977 s`.
This is a 4.96x failure at the median and no repetition passed.

This is not presented as a mathematical lower bound on an exact-shape run.
It is a hard *procedural* terminal result: the frozen exact-shape draft requires
the tiny residual prerequisite to pass before a launch seal can exist.  The
prerequisite failed while exact bills, inventories, output identity, strided
`out=`, and D5 byte parity all passed.  Running the multi-gigabyte exact shape
would bypass the preregistered escalation rule and is therefore rejected.

## Frozen bindings

- D6 design authority:
  `D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md`, SHA-256
  `808a41b8fa1429e1efaf978cd3f17f18ecf7ae7f21c20a9359546922b3696da7`.
- Independent D6 design audit:
  `INDEPENDENT_REDTEAM_D6_SUCCESSOR_AUDIT.md`, SHA-256
  `df76169237166cf685ca9e0474643d1d668dcc2ae762f5e9e57af0e6b8afd6f2`.
- Candidate source: `estimator.py`, SHA-256
  `e9ebe9abebe4d632e20d81396aea2bc64a511277d88bc6fd13fb6a80721ffd58`.
- Accounting/liveness manifest: SHA-256
  `bced5170098ccef0a702414a212dea606ef3c16e1c823c2ed2973888d5f0558c`.
- Static verifier/result: SHA-256
  `b6cef81fab76d6ca63a47554a79c5ba4970c26fda614ebc738a5f156a6d3fcd1` /
  `4756f8d22b032d3a96058caafab92ea18d3e76bb64a763642636b82944af2fe7`.
- Serialization verifier/result: SHA-256
  `09b6b953f4386c6ba213f221bd595ff107ad39d2ec62ce6128b72557f2502a1b` /
  `9a3ffa19dab76b59443f86b2dbace7a1e2c172b66f3791f27d75d960fc01b45b`.
- Sealed isolated tiny authority final seal / receipt / kernel / final verifier:
  SHA-256 `2f2f8d26a4892b727ce7be43664bcf001e48fa61f35ff184b98f1d6f891b85ed`,
  `fcddb6013402ae896aeda204705be543b6811c747bd79092c5feb99b417ae20c`,
  `a09710a67b0f2e571b82b62909d5f699f2993507d7bd8c739d30c8d0d397a9f1`,
  and `292fdb157c45eef934815e62aec858c841758a86ca0e1c84599af84766fb25db`.
- Same-source independent tiny residual script / receipt: SHA-256
  `4e1f58c73d5d7d7fcc1622fdfb0c62f4f1d495f4f5d67a0a8ab8f00e49d45102` /
  `c08ce3f9a6ed054f2a3c5a8942ca0e4b93970ba5987d4699c6171765538ca7f8`.

No estimator was imported or called by this audit or either tiny authority.
No fixture, exact-shape operand, target, score, name, board, common driver,
submission, or network resource was touched.

## Exact physical arena and liveness

The sealed one-arena formula is

`sum(j=1..4) (7/4)^j(A+B) + (7/4)^3 max(A,B) + (7/4)^4 C + 3(7/4)^3 C + 2C`.

It is implemented as 15 contiguous, nonoverlapping physical regions.  The
following intervals are float32 element offsets, half-open `[start, stop)`:

| physical region | push interval | read interval | live use |
|---|---:|---:|---|
| operand A level 1 | `[0, 1,835,008)` | `[0, 25,690,112)` | forward level 1 onward |
| operand B level 1 | `[1,835,008, 29,360,128)` | `[25,690,112, 51,380,224)` | forward level 1 onward |
| operand A level 2 | `[29,360,128, 32,571,392)` | `[51,380,224, 96,337,920)` | forward level 2 onward |
| operand B level 2 | `[32,571,392, 80,740,352)` | `[96,337,920, 141,295,616)` | forward level 2 onward |
| operand A level 3 | `[80,740,352, 86,360,064)` | `[141,295,616, 219,971,584)` | forward level 3 onward |
| operand B level 3 | `[86,360,064, 170,655,744)` | `[219,971,584, 298,647,552)` | forward level 3 onward |
| operand A level 4 | `[170,655,744, 180,490,240)` | `[298,647,552, 436,330,496)` | leaf input |
| operand B level 4 | `[180,490,240, 328,007,680)` | `[436,330,496, 574,013,440)` | leaf input |
| S/T dependency scratch | `[328,007,680, 412,303,360)` | `[574,013,440, 652,689,408)` | one transform dependency generation at a time |
| leaf products | `[412,303,360, 559,820,800)` | `[652,689,408, 662,523,904)` | leaf through first reconstruction |
| reconstruction bank 0 | `[559,820,800, 644,116,480)` | `[662,523,904, 668,143,616)` | level 3, then reused for level 1 |
| reconstruction bank 1 | `[644,116,480, 728,412,160)` | `[668,143,616, 673,763,328)` | level 2 only |
| U2/U3/U4 scratch | `[728,412,160, 812,707,840)` | `[673,763,328, 679,383,040)` | one reconstruction level at a time |
| output bank 0 | `[812,707,840, 828,436,480)` | `[679,383,040, 680,431,616)` | first sequential result |
| output bank 1 | `[828,436,480, 844,165,120)` | `[680,431,616, 681,480,192)` | second sequential result |

The push layout is the maximum at `844,165,120` elements, or
`3,376,660,480` bytes.  The read layout needs `681,480,192` elements but is
bound into the same flat allocation.  The 26 push and 26 read logical views
are contained in their named owner intervals.  Logical overlaps are safe:

1. At each forward level, the four S dependencies are computed, copied into
   the disjoint A destination, and dead before the same dependency interval is
   overwritten by T.  T is then copied into the disjoint B destination.
2. The leaf destination is disjoint from both transformed operands.
3. Reconstruction follows `leaf -> bank0(level3) -> bank1(level2) ->
   bank0(level1) -> out`; bank 0 is not overwritten while it is a source.
4. All four U views share one physical region but are used at distinct unwind
   steps.  Within a step U2, U3, and U4 occupy distinct leading slices.
5. Read channel 0 writes output bank 0.  Channel 1 reuses only the work regions
   and writes output bank 1, so channel 0 survives until the final combination.
6. Push channels write directly to distinct alternating CP-state destinations;
   the source and destination CP banks are disjoint.  Push's conservative arena
   output slices remain unwritten.

No live spatial collision was found.  There is also no hidden source/destination
overlap in any `add`, `subtract`, `copyto`, or leaf `matmul` call that would
require an unaccounted full-size temporary.

## Iterative branch and reconstruction order

The forward transform reproduces the frozen D5 law exactly:

- left dependencies: `s1=a21+a22`, `s2=s1-a11`, `s3=a11-a21`,
  `s4=a12-s2`;
- left branch tuple: `(a11,a12,s4,a22,s1,s2,s3)`;
- right dependencies: `t1=b12-b11`, `t2=b22-t1`, `t3=b22-b12`,
  `t4=t2-b21`;
- right branch tuple: `(b11,b21,b22,t4,t1,t2,t3)`.

Each new transform writes `destination[child]` in front of all existing parent
axes.  Therefore the flattened leaf index is `child*parents + parent`; after
four levels its C-order path sequence is exactly `np.ndindex((7,7,7,7))`.
The sealed tiny proof covers all 2,401 paths and hashes that sequence as
`a32d9b1e570b05f0de01ede892bd26aec3a520723826c836be6fdde1112c6281`.

Unwind is level 4 to level 1 in the frozen association order:

`c11=p0+p1; u2=p0+p5; u3=u2+p6; u4=u2+p4; c12=u4+p2;
c21=u3-p3; c22=u3+p4`.

This is byte-identical to the recursive D5 spelling on both the sealed maximum
tiny case and an active-below-capacity case.  The latter additionally proves
that non-C-contiguous A/B leaves, leaf `matmul(out=...)`, and final output work;
inactive guard columns remain untouched and the first output survives the
second channel.

## FlopScope semantics and billed-operation conservation

The authoritative environment is the `data_gate/.venv` environment with
FlopScope `0.12.0+np2.4.6` and NumPy `2.4.6`.  Relevant source hashes are:

- `_array_ops.py`: `e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c`;
- `_pointwise.py`: `7d376515446f1b174bef375c48116a552098bd514f5552450876f56822beb0c7`;
- `_validation.py`: `6a047b60aaa5954b9301a15a6e3e5f646d10a95ace60c6b47c54cd9f2b52531f`;
- `_einsum.py`: `0f25b8f5709c5a96404f20c38afc9b705568ab2654ab6c5fc81f3de292b73284`.

`fnp.reshape` charges the complete input element count even when NumPy can
return a view.  `fnp.copyto` charges one operation per destination element.
Pointwise `out=` and strided `fnp.matmul(..., out=...)` preserve the supplied
destination object.  Thus the 52 cold persistent view bindings are not free:

- push: 26 reshape calls, `986,193,920` FLOPs;
- read: 26 reshape calls, `933,404,672` FLOPs;
- total: 52 calls, `1,919,598,592` FLOPs.

One channel's exact hot inventory is `copyto=56`, `subtract=32`, `add=28`,
`matmul=1`: 117 wrapper operations.  At transform level `l`, this conserves
`11(A_l+B_l)` billed element operations; reconstruction conserves `7C_l`.
The leaf is one 2,401-way batch matmul.  Summed over the exact 58-channel
roster:

- dense direct roster: `966,086,623,232` FLOPs;
- D5 materialized roster: `594,693,332,992` FLOPs;
- D6 arena roster: `591,553,830,912` FLOPs;
- direct-quadrant reconstruction saves `3,139,502,080` billed copies;
- sequential common-left handling saves another `20,971,520` billed copies.

After charging both cold view maps, projected D6 calls are
`[1,319,635,780,983, 1,356,359,358,984, 1,317,713,029,632]`.  The projected
incumbent-minus-D6 margins are
`[372,849,123,328, 374,768,721,920, 374,768,721,920]`, all at least the required
`371,608,248,320`.  Cold-call headroom over the minimum is only
`1,240,875,008`; omitting reshape billing would have been a material error,
but the frozen accounting includes it.

## Memory premise

The sealed memory formula is internally feasible; this audit found no proof of
impossibility.  The incumbent anchor plus push arena is `6,798,163,968` bytes,
leaving `1,791,770,624` bytes below 8 GiB.  The drafted synthetic smoke caps
incremental RSS at `3,913,531,392` bytes and absolute RSS strictly below 7 GiB.
Those figures are arithmetically compatible with the arena and anchor.

That is a static admission result, not a measured exact-shape RSS result.  Since
the residual prerequisite failed, exact-shape RSS remains intentionally
unmeasured and no memory success claim is promoted.

## Residual-wall finding

`run_independent_tiny_residual_diagnostic.py` parses and compiles only the
candidate's three D6 kernel helpers.  It builds a tiny arena independently from
the sealed formula, validates a gapped strided call byte-for-byte against
recursive D5, warms both spellings, then alternates eight measurements.  Every
measurement contains the full 58-call aggregate.  Threads are fixed at two.

| aggregate arm | residual samples (seconds) | median |
|---|---|---:|
| D6 same-source arena | `0.0504188500, 0.0499564679, 0.0524143088, 0.0498867994, 0.0499045127, 0.0529015046, 0.0499511720, 0.0501312828` | `0.0500438754` |
| direct `fnp.matmul(out=...)` | `0.0004483778, 0.0004177927, 0.0004619956, 0.0004007565, 0.0004323763, 0.0004154518, 0.0004650000, 0.0004355353` | `0.0004339558` |
| paired D6 minus direct | `0.0499704722, 0.0495386752, 0.0519523132, 0.0494860429, 0.0494721364, 0.0524860527, 0.0494861719, 0.0496957474` | `0.0496172113` |

All D6 reps had the exact aggregate inventory (`add=1,624`, `copyto=3,248`,
`matmul=58`, `subtract=1,856`) and exact bill.  All direct reps had 58 matmuls
and their exact bill.  The failure is therefore not caused by an accidental
extra metered numerical operation.  It is the dimension-insensitive residual
of 6,786 wrapper calls plus Python slicing/loop dispatch versus 58 direct
wrappers.  Larger matrices can change backend time, which FlopScope subtracts;
they do not remove this fixed control surface.

## Authorization boundary

- Authorized scientific result: the already-completed sealed tiny semantic
  authority and this read-only independent audit.
- Rejected: creation of an exact-shape launch seal for this D6 form.
- Rejected: exact-shape execution, estimator import/call, fixture call, common
  driver, target/score/name/board access, production, submission, or network.
- Reopening condition: a materially different kernel spelling must reduce the
  fixed wrapper/control surface and independently pass a newly sealed tiny
  residual prerequisite.  Changing only the recursion depth, arena dimensions,
  or timing sample is not a valid retry of this closed form.

