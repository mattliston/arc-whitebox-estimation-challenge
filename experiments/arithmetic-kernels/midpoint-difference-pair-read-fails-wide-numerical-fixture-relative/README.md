# Midpoint/difference pair read fails wide numerical fixture: relative error 0.0035854277148522043

2026-09-21 · experiment · rejected · measured

The exact real-arithmetic signed-pair identity retained the contracted width and original product kernel but changed float32 association.

Recorded measurement:

    prospective wide-fixture tolerance. Relative error is **0.0035854277148522043**
    against **0.00003**, about **119.5 times** the limit. This closes the fixed
    331684: adjusted 3.5844144212297072e-9, raw 1.949715441540434e-8,
    100/100 cases complete, zero failures. The 4e-9 milestone is banked;
    3e-9, 2e-9 and approximately 1e-9 remain unmet. The full goal stays active.
    identity a*(x*x*u-y*y*v) = (h*h+d*d/4)*(a*D)+(2*h*d)*(a*H),
    where h=(x+y)/2, d=x-y, H=(u+v)/2 and D=u-v. It packs the same
    is FP32. The common old basis, lazy map and addition order are retained.
    Marginal cubic diagonals remain on their existing FP64 path. Factor
    the dense FP64 reference left a 5.679343126843894e-16 roundoff norm.
    unit_v2 handles that declared exact-zero case, bounds the dense roundoff,
    The repaired unit then failed on the fixed 512x2048 by 2048x512 close-pair
    arithmetic paths below. Its FP64 polarization identity agrees with direct
    FP64 evaluation to relative 5.478158045314982e-13.
    | Pair rewrite, existing kernel | 0.003585427714852204 | 730,838,014 |
    | Original signed read, existing kernel | 0.01217643682958431 | 675,172,352 |
    | Pair rewrite, direct matmul diagnostic | 4.206151103946926e-07 | 1,082,919,934 |
    | Original signed read, direct matmul diagnostic | 0.0003895404827650046 | 1,075,576,832 |
    Four guarded groups retired; controllers 5738, 89525, 17099 and 51028 are
    terminal. Peak sampled group RSS was 71,352,320
    271,073,280 bytes. Failed unit processes did
    nice10, one BLAS thread and the existing 8GB group limit. About
    0.707 MB was preserved before

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/polarized_pair_read_native_v1/candidate.py](code/polarized_pair_read_native_v1/candidate.py)
- [code/polarized_pair_read_native_v1/common.py](code/polarized_pair_read_native_v1/common.py)
- [code/polarized_pair_read_native_v1/diagnose_wide.py](code/polarized_pair_read_native_v1/diagnose_wide.py)
- [code/polarized_pair_read_native_v1/diagnose_zero_reference.py](code/polarized_pair_read_native_v1/diagnose_zero_reference.py)
- [code/polarized_pair_read_native_v1/finalize.py](code/polarized_pair_read_native_v1/finalize.py)
- [code/polarized_pair_read_native_v1/native.py](code/polarized_pair_read_native_v1/native.py)
- [code/polarized_pair_read_native_v1/native_v2.py](code/polarized_pair_read_native_v1/native_v2.py)
- [code/polarized_pair_read_native_v1/preflight.py](code/polarized_pair_read_native_v1/preflight.py)
- [code/polarized_pair_read_native_v1/prepare.py](code/polarized_pair_read_native_v1/prepare.py)
- [code/polarized_pair_read_native_v1/production.py](code/polarized_pair_read_native_v1/production.py)
- [code/polarized_pair_read_native_v1/qualify.py](code/polarized_pair_read_native_v1/qualify.py)
- [code/polarized_pair_read_native_v1/run.py](code/polarized_pair_read_native_v1/run.py)

<details>
<summary>Additional programs and dependencies (6)</summary>

- [code/polarized_pair_read_native_v1/run_v2.py](code/polarized_pair_read_native_v1/run_v2.py)
- [code/polarized_pair_read_native_v1/score.py](code/polarized_pair_read_native_v1/score.py)
- [code/polarized_pair_read_native_v1/unit.py](code/polarized_pair_read_native_v1/unit.py)
- [code/polarized_pair_read_native_v1/unit_v2.py](code/polarized_pair_read_native_v1/unit_v2.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L4](RESULT.md#L4)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L7](RESULT.md#L7)
- [RESULT.md#L8](RESULT.md#L8)
- [REEXAMINATION_INDEX_20260919.md#L554](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L554)
- [REEXAMINATION_INDEX_20260919.md#L557](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L557)

<details>
<summary>All result files and records (6)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-midpoint-difference-pair-read-fails-wide-numerical-fixture-relative) · [Copy provenance](../../../PROVENANCE.json)

