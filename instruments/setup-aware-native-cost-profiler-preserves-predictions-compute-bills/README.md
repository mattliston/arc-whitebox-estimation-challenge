# Setup-aware native cost profiler preserves predictions and compute bills while attributing call stacks

2026-09-26 · tool · informational · measured

The profiler retained every original charge and invoked the unchanged native evaluator while recording call stacks. Setup charges were excluded from the prediction denominator, and profiling overhead was excluded from runtime comparisons.

Recorded measurement:

    Ranks35/100 all prediction bytes and bills exact;old/young transport47.18%,plane read33.34%,covariance/star11.62%;inclusive sites overlap.

## Programs

- [code/profile_candidate.py](code/profile_candidate.py)

## Evidence

- [NEXT_RESEARCH.md#L29](../../experiments/arithmetic-kernels/batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md#L29)
- [NEXT_RESEARCH.md#L30](../../experiments/arithmetic-kernels/batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md#L30)

<details>
<summary>All result files and records (3)</summary>

- [NEXT_RESEARCH.md](../../experiments/arithmetic-kernels/batched-leaf-multiplication-ordinary-matrix-multiplication-faster-than-tensor/NEXT_RESEARCH.md) · record
- [s30_profile_summary_v1.json](s30_profile_summary_v1.json) · result
- [s30_profile_train2_v1_profile.json](s30_profile_train2_v1_profile.json) · result

</details>

[Research graph](../../graph/index.html#node=2026-09-26-setup-aware-native-cost-profiler-preserves-predictions-compute-bills) · [Copy provenance](../../PROVENANCE.json)

