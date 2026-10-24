# Fixed k32-mod64 padded-PUSH source preparation

2026-09-10. Only source preparation is selected now. No build, estimator,
runtime, target read, fitting, sweep, native expansion or submission has run
or is authorized by the existence of these files. Root separately reviews
and launches any explicit stage. Preserve APUSH/CD and all old failures.

## Exact parent and fixed extension

Parent is current_comp_aligned_push_d6_v1/build_v1/estimator.py,
SHA33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a.
Bind its actual three-seal build and two-context runtime results headed by
f841d13473244938a24de14822529960455a2f344306d2899c6656fed4e44f0a and
5c1cb763e99882197d6c0fb6fdf5dc2e7e23b6ae100ac1d16ff60e5cf18b5d9e.
They establish that exact source and a limited TRAIN35 runtime/fidelity
PASS, not the future padded source,37 runtime or any adjusted-score result.

Keep the original aligned-PUSH branch byte-for-byte. Add only original
FP32 PUSH-owner m1024,0<k<=1024,k%64=32,1024<n<=8192,n%64=0, with compatible
B/out, to serial D6 after padding k to k+32. No READ, square, recursive child
dispatch, model, learned literal, support threshold, rank, dtype or kernel
arithmetic changes. The seven D5 child calls still bypass the top wrapper.
Padding preserves AB in exact real arithmetic; changed FP32 summation is
not presumed byte-exact or raw-accuracy-neutral.

The builder makes exactly three additions: one cold padding binding call
immediately after the old serial-layout binding, one final-dispatch branch,
and one inert appended helper block. Original imports, learned literals,
all other definitions and every unrelated byte remain unchanged. The whole
source inverse must recover the exact parent bytes. Entire-AST comparison
also admits only the two declared functions and new inert metadata/helpers.
Cached UTF8 byte/line offsets extract AST spans without repeated scans of
the large learned literal. No estimator module is imported or evaluated.

## Complete scratch ownership and lifetime

The old serial-PUSH maximum top+child views end at arena element118006528.
The exact original arena has292225024 FP32 elements (1,168,900,096 bytes).
Two disjoint new basic views use:

    A_pad [118006528,119055104), shape(1024,1024)
    B_pad [119055104,127443712), shape(1024,8192).

The builder checks every old serial view lies at or below118006528, both
new spans are contiguous/disjoint, and their end is below the original
arena limit. There is no second full backing buffer. Other scratch roles
may reuse/overwrite these locations between calls; no old value in either
pad is relied upon. Every invocation copies all live A/B elements and
explicitly rewrites BOTH32-wide zero strips before entering serial D6.

The helper requires the original _SCRATCH[1] owner, matching existing
serial binding/arena, its own padding busy flag initially false, and the
old serial busy flag initially false. A try/finally releases the padding
flag even if the old kernel fails. The unchanged serial try/finally owns
its own busy flag and directly calls the original D5 helper. The pads are
not referenced by those child/top view regions and remain stable until
serial return. The caller's original output is passed directly through;
no padded output, post-copy or caller array is retained.

Source callsite proof excludes both old square callers by _SCRATCH[4]
identity. The only admissible final _mm caller allocates its own output
outside scratch; its input operands are the unchanged external CP/weight
consumer values, recursively sliced by _mm, never a scratch-frame output.
The original packed/square helpers return caller-owned results. This
source invariant is essential: no arbitrary caller whose input aliases
the padding spans is admitted. No new public arbitrary-matrix entrypoint
or alternate scratch owner is introduced. Later qualification must retain
these source/owner checks rather than generalize the policy.

Only the two bounded shape views, their arena owner, and a scalar FP32 zero
persist. The zero is created with public fnp.zeros((),dtype=fnp.float32)
inside the first official context, never at import. It prevents unintended
float64 scalar billing in clear-copy calls. Scratch contents are temporary
workspace, not a hidden carried state; all numerical/observational owners
must retire with the existing process/context lifecycle.

## Exact complete price and resource caveat

The existing _cp3_bind_arena_views makes two additional public reshapes,
charging1,048,576+8,388,608=9,437,184F once. Scalar zeros costs0F but is an
actual public call. Complete cold initialization becomes1,954,407,482F,
not1,944,970,298F. Both old serial layout sets and all other initialization
remain paid; no prefix or nested bill is counted twice or subtracted.

Each new tile pays the unchanged P6(1024,k+32,n) tariff PLUS

    k*(1024+n) copied input elements + 32*(1024+n) zero-strip writes.

There are exactly four additional FP32 copyto calls per tile, no extra
output charge. Public reshape/copy/zero source and the frozen D6 tariff
owner are pinned. All other complete card work remains paid unchanged.

The parent PAD32_PRICE_20260910.md source/metadata estimate is preserved:
514 new calls across40 (303 full,211 tails), maximum22 on rank37;
450,955,876,627.575 ->438,679,353,440.350 projected meanF, or2.72233356%
below projected APUSH. This is not an actual new bill, raw score or expected
milestone gain. New rounding can change later masks and full cost.

No enlarged backing array does not imply unchanged measured RSS: touched
pages, view handles and bounded shape caches still matter. Across40 the
combined D6 shapes number151, exceeding the128-entry frame cache. Each new
tile replaces73 public D4 calls by655 D6+4 copy calls, adding586 operations;
mean7,530.1 and maximum12,892 additional public calls/network. Runtime and
residual must be measured, not inferred from lower FLOPs or old READ timing.

## Inert builder and future fixed runtime falsifier

Only this explicit source-stage command may emit build_v1/estimator.py:

    OFFPY -B -u build.py --run --source-sha256 REVIEWED --protocol-sha256 REVIEWED

Use nice10, assertions and official OFFPY; no numerical imports. The builder
has an internal60s/<2GB whole-stage guard, exclusive never-resumed output,
exact source readback and unconditional final direct-inventory rehash.
Result/completion/finalization are exclusive actual three seals with digest
links and passed/completed flags. Any failure retains partial artifacts and
false seals; no retry or source repair occurs automatically. Finalization's
elapsed timestamp follows completion I/O; its own publication/hash and exit
also must finish below60s. Independent external supervision is separately
reviewed and required before any runtime; builder success is source-only.
The inert derive(tick) API returns source text/proof without executing it.

If root selects the subsequent falsifier, use ONE unchanged original
NumPy-free persistent client/server, canonical TRAIN37 cold then repeat37.
Original client check_remote_series_v1.py indexes manifest row37; that
entry is role_rank37, rows/observer_fit/0037/row_manifest.json, and weight
SHA967d989cf0ddf4770f199d50c97b26c70b1129d2d739929e82c8e229b7fd43d4,
the exact same physical fixture as current native CHECK4. This builder
checks metadata mapping only and does not read the weights or write a client
adapter. Runtime must authenticate the actual weight bytes before use.

Runtime gates are unchanged: each own complete context<60s, residual<.4s,
F<=2^41 with exact multiplier, cold/repeat F difference1,954,407,482, finite
16x1024 FP32 output, own byte-exact repeat and all-layer maxabs drift<=1e-5
against separately sealed same-fixture parent native output. Both own bills
must improve over parent with the same input convention and cold ownership.
No native-wall number is treated as a NumPy-free production timing control.
Independent whole-stage300s/next-completed-context120s, combined8GB, zero
added swap, actual completed samples<=1s, no foreign worker, nice10/BLAS1,
all owned groups reaped and actual three seals/exit0 are mandatory. A failed
gate stops the extension; no shape retry, roster swap or gate relaxation.
No such runtime adapter, process or result is created in this source task.
