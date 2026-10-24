# Independent red-team: D5R1 closure and D6 successor selection

Date: 2026-08-26

Disposition: **AUTHORIZE_PREPARATION_ONLY for P2R0C1D6.** The sole promoted
design is a sequential-channel, zero-stream, fully preallocated depth-4
Strassen--Winograd arena. This receipt does not authorize an estimator call,
an exact-shape smoke, a common-driver launch, target access, production, or a
submission. It does not edit the canonical graph.

`P2R0C1D5R1` remains terminally closed with no retry. `P2R0C1D6` is open only
as an unimplemented and unmeasured design. There is not yet an external
constraint: one materially different resource schedule remains plausible.
If its frozen implementation cannot prove its arena or fails its separately
authorized target-free exact-shape smoke, close this Strassen resource lane
absent a native/custom backend rather than reopening D5 or a streamed variant.

## Evidence bound by this review

| Artifact | SHA-256 |
|---|---|
| `D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md` | `808a41b8fa1429e1efaf978cd3f17f18ecf7ae7f21c20a9359546922b3696da7` |
| `verify_d5r1_resource_successor.py` | `6214a1e493a9c543815bbd9f3020023e11021f369449ba815715346cc1bcbe5a` |
| D5 `estimator.py` | `69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5` |
| incumbent `public_noeigh_card_v1/estimator.py` | `373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7` |
| `RECOVERED_REFERENCE_EVIDENCE.json` | `cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6` |
| `RECOVERY_DESCENDANT_RESULT.json` | `a8221d116c03ed442cb6343a54b429b06e5a620f905c056fa99abe6497bc9549` |
| `RECOVERY_DESCENDANT_TERMINAL_RECEIPT.json` | `8ee4d2be92f780b4f547e2129e59241cb19936a9767df1f04b3e9b6ba9b281c0` |

The upstream verifier was rerun read-only and printed
`PASS_D5R1_RESOURCE_CAUSAL_AND_D6_DESIGN`.

## Terminal D5R1 facts

Exactly one candidate call, ordinal 0 / role-rank 35, completed before the
no-retry stop. Its literal bill was `1,320,876,655,991`, exactly
`371,608,248,320` below the measured incumbent. Float32 final-output drift was
small (RMS `4.109649235540117e-8`, maximum absolute
`2.384185791015625e-7`), but the candidate failed both decisive resource
gates: residual wall was `1.875688566826284` seconds and peak RSS was
`11,953,569,792` bytes. Rank 279 and the repeated rank 35 were never called.
No raw MSE was measured.

The operation ledger supports the causal diagnosis. Relative to the same
ordinal's reference, D5 added `1.4931683861650527` seconds of residual wall.
Its `subtract`, `stack`, `add`, and `block` group consumed
`12.20690989587456` backend seconds versus `0.4394297036342323` seconds in
the reference. The static peak missed literal RSS by `3,916,464,128` bytes.
The nested dynamic `stack`/`block` trees and paired-channel materialization,
not the depth-4 arithmetic identity itself, are the failed mechanism.

The three immutable incumbent residuals were
`[0.3825201806612313, 0.3858333397656679, 0.37958328472450376]` seconds.
The worst strict-0.4 margin is only `0.0141666602343321` seconds. Relative to
D5's measured incremental residual, a successor must eliminate at least
`99.05123491994652%` of that increment at the worst ordinal. This makes the
wall case narrow and empirical; an arena is plausible, not proven.

## Why zero-stream D4 is the only promoted schedule

The frozen CP3 inventory is 30 push plus 28 read channel contractions, with
the final K3 matvec excluded. Sequential channels therefore give exactly 58
top-level kernels and 58 leaf-batch `matmul` dispatches, each leaf batch
holding `7^4 = 2,401` products. D5 used 37 top calls by batching selected
channel pairs, so D6 pays only 21 additional leaf dispatches while removing
the paired tree and every recursive large allocation.

Streaming is a worse wall-risk trade at the admitted memory point:

| schedule | leaf-batch matmul calls | disposition |
|---|---:|---|
| zero-stream D4, sequential channels | 58 | promote for preparation |
| stream top one level, batch D3 below | 406 | close as separately promoted successor |
| stream top two levels, batch D2 below | 2,842 | close as separately promoted successor |
| stream every leaf | 139,258 | deduped/closed |

The streamed forms retain the same arithmetic but multiply Python and meter
dispatches exactly where only 14.17 ms of strict residual margin remains.
They are not needed to fit the conservative D6 arena, so they do not buy an
admission benefit that offsets their wall risk.

Depth reduction is also not promoted. Under the already audited D5
materialization accounting, depths 3, 2, and 1 retain respectively
`304,359,964,672` (81.90%), `219,690,172,416` (59.12%), and
`118,459,203,584` (31.88%) of D5's exact saving. More importantly, merely
scaling D5's incremental residual by the recursive allocation-volume ratios
still predicts call-0 totals of about 1.159, 0.750, and 0.516 seconds. This is
a plausibility screen rather than a lower bound, but it shows that reducing
depth without replacing the allocator does not target the observed failure.

## Independent memory check

For `q=7/4`, one channel, and `A=m*k`, `B=k*n`, `C=m*n`, the conservative
arena admission is

`sum(j=1..4) q^j(A+B) + q^3 max(A,B) + q^4 C + 3q^3 C + 2C`.

This is deliberately the larger previously checked recursion-capacity bound,
not the tighter alternating-bank liveness estimate. At the worst push
`(1024,1024) @ (1024,15360)`, it is exactly `844,165,120` float32 elements,
or `3,376,660,480` bytes. Adding the incumbent anchor `3,421,503,488` gives
`6,798,163,968` bytes (6.331 GiB): `1,791,770,624` bytes below 8 GiB and
`718,028,800` bytes below 7 GiB. The worst read is smaller at
`681,480,192` elements and `6,147,424,256` bytes including the anchor.

This arithmetic is sufficient for preparation, but it is not an RSS proof.
D5's 3.916 GB model miss must not be added as a reusable constant or silently
ignored. D6 must statically prove the old allocation tree is absent, bind
nonoverlapping slices for every live region, and then measure fresh. The
proposed smoke cap of arena plus 512 MiB is `3,913,531,392` bytes of process
increment; anchor plus that cap is `7,335,034,880`, only `181,157,888` bytes
below 7 GiB. Therefore both conditions are required independently: increment
no greater than that cap and literal absolute RSS strictly below 7 GiB.

## Accounting, numerical, and residual conditions

The arithmetic roster and Winograd association may remain depth 4, so losing
the `371,608,248,320` saving is not inherent to D6. Eliminating the paired
common-left stack may reduce the bill further, while explicit copies and
reconstruction materialization can add counted work. Projection is
insufficient: a frozen literal meter must prove candidate bills no greater
than D5's exact `[35,279,35]` trio
`[1,320,876,655,991, 1,359,519,832,584, 1,320,873,503,232]` and measured
incumbent-minus-candidate deltas at least `371,608,248,320` on every call.
Every large `out=` shape and overlap assumption must be exercised before a
common-driver launch; unsupported `out=`, a hidden full-size temporary, or
dummy billed work is a terminal static/smoke failure.

A separately sealed target-free exact-shape smoke must compare the complete
58-contraction roster with direct incumbent spelling and require residual
delta `<= 0.010` seconds, in addition to the memory conditions above. That is
only an admission screen. It leaves roughly 4.17 ms against the worst strict
0.4 reference margin and does **not** certify the later deployment gate
`<= 0.385`: the recovered rank-279 incumbent itself measured
`0.3858333397656679`. Only a later, independently authorized, sealed
zero-reference-call `[35,279,35]` common-driver gate can resolve absolute
residual, all-roster drift, repeat parity, and literal RSS.

## Dedupe boundary

The Fable classic S3 sources, prior recursive Winograd S3, and D5 all use
recursive dynamic `stack`/`block` materialization; D6 instead applies an
iterative fixed arena to the exact CP3 roster. `K_probe_strassen.py` tests
local assembly substitutions but not a full CP3 four-level arena. The Phase-1
TD52/Strassen artifacts use high-rank singleton `tensordot` leaves on the
sample-forward path, a mechanism explicitly forbidden by the D5/D6 FlopScope
scope. The native Winograd probe is a packed sampling backend with different
shapes and objective; its depth-3 wall regression warns against assuming
arithmetic savings imply speed, but it is not this resource schedule.

Thus D6 is materially new, while top-level/per-leaf streaming, depth-only
retunes, TD52 composition, and D5 threshold or batching retunes are not
independent successors to promote now.

## Graph-ready status

- `P2R0C1D5R1`: `closed`, terminal
  `CLOSE_THIS_IMPLEMENTATION_FORM_NO_RETRY`.
- `P2R0C1D6`: `open`, design-only, unimplemented, unmeasured;
  `AUTHORIZE_PREPARATION_ONLY`; no run is active or authorized.
- `P2R0C1D`: remains `open` because D6 deployability and the broader
  accuracy-compute route remain unresolved.
- `active_work`: empty.

