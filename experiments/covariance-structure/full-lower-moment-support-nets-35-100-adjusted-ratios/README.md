# Full lower-moment support on nets 35/100: adjusted ratios 0.99071/0.96585 fail the 5%-each gate (pooled 2.1668% better)

2026-09-20 · experiment · rejected · measured

The same candidate (full covariance, repeated third-moment star target and trace vector b retained with natural-order full reads; CP masks, births, thresholds, rank 384, age 4, schedule and all learned parameters fixed) was scored under a separately pre-sealed objective protocol: it improves both development networks, with raw ratios 0.9614/0.9375 at a 3.05%/3.03% higher cold bill, but misses the fixed requirement of at least 5% adjusted improvement on each network. The record classifies this as insufficient gain, not a worse-score result; the candidate later passed the broader panels and became #331672.

Recorded measurement:

    net 35: parent raw 1.87060284901e-8, candidate 1.79839117567e-8, cold 410218730507 F, raw ratio 0.961396576844, adjusted 0.990710344893; net 100: parent 1.84245341753e-8, candidate 1.72722521502e-8, cold 413034826010 F, raw ratio 0.937459367272, adjusted 0.965851998783; pooled adjusted 0.9783316543482492 (2.1668%), pooled raw 0.9495187084747343; cost +3.0491%/+3.0287%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/full_lower_moment_support_v1/candidate.py](code/full_lower_moment_support_v1/candidate.py)
- [code/full_lower_moment_support_v1/build.py](code/full_lower_moment_support_v1/build.py)
- [code/full_lower_moment_support_v1/run_native.py](code/full_lower_moment_support_v1/run_native.py)
- [code/full_lower_moment_support_v1/score.py](code/full_lower_moment_support_v1/score.py)
- [code/full_lower_moment_support_v1/unit.py](code/full_lower_moment_support_v1/unit.py)
- [code/full_lower_moment_support_v1/common.py](code/full_lower_moment_support_v1/common.py)
- [code/full_lower_moment_support_v1/helpers.py](code/full_lower_moment_support_v1/helpers.py)
- [code/full_lower_moment_support_v1/null.py](code/full_lower_moment_support_v1/null.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)
- [code/weight5_reuse_v1/native.py](code/weight5_reuse_v1/native.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L9](NEGATIVE_RESULT.md#L9)
- [NEGATIVE_RESULT.md#L10](NEGATIVE_RESULT.md#L10)
- [NEGATIVE_RESULT.md#L12](NEGATIVE_RESULT.md#L12)
- [NEGATIVE_RESULT.md#L13](NEGATIVE_RESULT.md#L13)

<details>
<summary>All result files and records (3)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · result
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · record

</details>

## Related work

- enables: [Submission #331672: full lower-moment support (full covariance, star and trace reads) added to the lazy-rotation parent](../../../submissions/331672/README.md)
- ← measured by: [Submission #331672: full lower-moment support (full covariance, star and trace reads) added to the lazy-rotation parent](../../../submissions/331672/README.md)

[Research graph](../../../graph/index.html#node=2026-09-20-full-lower-moment-support-nets-35-100-adjusted-ratios) · [Copy provenance](../../../PROVENANCE.json)

