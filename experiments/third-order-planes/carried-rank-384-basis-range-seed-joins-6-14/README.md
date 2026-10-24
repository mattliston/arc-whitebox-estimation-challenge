# Carried rank-384 basis as the range seed at joins 6..14: pooled adjusted ratio .9962384901 fails the .995 gate

2026-09-19 · experiment · rejected · measured

The shared rank-384 join recomputes a one-pass range from the lambda-squared Gram using 384 columns of the next weight matrix as the seed Omega; this variant keeps that seed at the first join (layer 5) and reuses the carried gated basis Qp as the seed at joins 6..14 with everything else unchanged. Net 100 improves but net 35 worsens, so the rule was closed without seed, blend or power tuning.

Recorded measurement:

    raw ratios 35 1.010312151, 100 .982156773; own-cold adjusted 1.010282289 / .982128063; pooled .9962384901; candidate cold F 412400805263 / 415319167336; candidate SHA d87a79fe582b33769497a5b9abc59786faf80d11c88d241edbc3a51a4637a739

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/warm_basis_cost75.py](code/warm_basis_cost75.py)
- [code/build_warm_basis.py](code/build_warm_basis.py)
- [code/score_warm_basis_dev.py](code/score_warm_basis_dev.py)
- [code/native_screen.py](code/native_screen.py)
- [code/cost_rule75_scalar.py](code/cost_rule75_scalar.py)

## Evidence

- [WARM_BASIS_RESULT.md#L9](WARM_BASIS_RESULT.md#L9)
- [WARM_BASIS_RESULT.md#L14](WARM_BASIS_RESULT.md#L14)
- [RECEIPTS.md#L311](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L311)
- [RECEIPTS.md#L313](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L313)
- [WARM_BASIS_PROTOCOL.md#L4](WARM_BASIS_PROTOCOL.md#L4)

<details>
<summary>All result files and records (3)</summary>

- [WARM_BASIS_PROTOCOL.md](WARM_BASIS_PROTOCOL.md) · record
- [WARM_BASIS_RESULT.md](WARM_BASIS_RESULT.md) · result
- [RECEIPTS.md](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-carried-rank-384-basis-range-seed-joins-6-14) · [Copy provenance](../../../PROVENANCE.json)

