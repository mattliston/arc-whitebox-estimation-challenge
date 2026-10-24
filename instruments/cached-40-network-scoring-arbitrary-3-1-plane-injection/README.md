# Cached 40-network scoring and arbitrary (3,1)-plane injection with verified native controls

2026-09-23T02:26Z · tool · informational · measured

The cached evaluator scores only the candidate against preserved parent rows. Plane injection controls reproduce the parent and direct plane doubling.

Recorded measurement:

    2026-09-22 19:26 porter: bank VALID40 cache fable_porter_v1/bank_valid40_cache.json + paired_eval_cached.py (cand side only, ~16 min). Generic (3,1)-plane plumbing build_t31_variant.py: off = bank bit-exact, X=0 bit-exact MSE, X=model K31 == direct T31 doubling (bit-identical MSE), transposed differs; cost 6n^2+4n flops + 6 ops per layer (+0.019% adj at L2-14). Service: eval_t31.sh <formula> <layers> <tag>. Contract: t31_formulas/TEMPLATE.py. Existing (2,1) read = 26% of the bill.

## Programs

- [code/fable_porter_v1/paired_eval_cached.py](code/fable_porter_v1/paired_eval_cached.py)
- [code/fable_porter_v1/build_t31_variant.py](code/fable_porter_v1/build_t31_variant.py)
- [code/fable_porter_v1/eval_t31.sh](code/fable_porter_v1/eval_t31.sh)
- [code/fable_porter_v1/t31_formulas/TEMPLATE.py](code/fable_porter_v1/t31_formulas/TEMPLATE.py)
- [code/fable_refit_v1/score_variant.py](code/fable_refit_v1/score_variant.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L19](../../records/TEAM_LOG_2026-09-22_to_09-26.md#L19)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log
- [bank_valid40_cache.json](bank_valid40_cache.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-09-23-cached-40-network-scoring-arbitrary-3-1-plane-injection) · [Copy provenance](../../PROVENANCE.json)

