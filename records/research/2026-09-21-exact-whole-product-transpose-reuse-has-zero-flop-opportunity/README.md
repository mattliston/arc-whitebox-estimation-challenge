# Exact whole-product and transpose reuse has zero FLOP opportunity on both 295-product traces

2026-09-21 · finding · rejected · derived

Canonical operand fingerprints and independent pairwise comparison found no repeated nonempty whole products.

Recorded measurement:

    All **295 outermost _mm products** on each of networks35 and100 have distinct
    products at layer0/1. Consequently the most favorable exact/transpose/zero
    whole-product reuse ceiling is **0 FLOPs, 0% of either complete cold bill**.
    The fixed5% implementation gate fails, and no cache candidate is created.
    | 35 | 413448696909 | 381662313492 | 295 | 0 |
    | 100 | 416264792412 | 384507304163 | 295 | 0 |
    Both observed forwards reproduce all16 original output bytes and complete
    cold bills. Every one of the295 ordered shape/dtype/caller/per-call bill
    The two unweighted Gram identities are already inside _resident_gram512 at
    layer0; their full combined bill is only859975168 FLOPs (about0.208%/0.207%).
    829713489321 FLOPs. Peak sampled group RSS was
    6317850624 bytes; self-reported native peak was
    6433816576 bytes. Zero added swap. About
    16.860 GB was read through
    1MiB payload blocks; no wide arrays or dataset copies were saved. No fitting,
    Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,
    100/100 complete,zero failures. The4e-9 milestone is banked;3/2/~1e-9 remain
    unmet. The full goal is active. The qualified0.41%-better local composition

## Programs

- [analyze.py](analyze.py)
- [audit_fingerprints.py](audit_fingerprints.py)
- [capture.py](capture.py)
- [common.py](common.py)
- [finalize.py](finalize.py)
- [identity.py](identity.py)
- [preflight.py](preflight.py)
- [prepare.py](prepare.py)
- [run.py](run.py)
- [unit.py](unit.py)
- [../2026-09-21-current-295-product-census-attributes-79-0853-80-2033/census.py](../2026-09-21-current-295-product-census-attributes-79-0853-80-2033/census.py)
- [../../../submissions/331684/estimator.py](../../../submissions/331684/estimator.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py](../../../experiments/learned-correction/learned-backward-adjoint-residual-observer-rejected-full-width-cross/code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L6](RESULT.md#L6)
- [RESULT.md#L7](RESULT.md#L7)
- [RESULT.md#L8](RESULT.md#L8)
- [REEXAMINATION_INDEX_20260919.md#L593](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L593)
- [REEXAMINATION_INDEX_20260919.md#L596](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L596)

<details>
<summary>All result files and records (6)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../../experiments/birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-exact-whole-product-transpose-reuse-has-zero-flop-opportunity) · [Copy provenance](../../../PROVENANCE.json)

