# Immutable common/difference base-factor cache: all outputs exact with 0.1158296% lower bill

2026-09-28 · experiment · adopted · measured

Three immutable join10 base-factor trees were copied into a bounded cache and retained through the lazy join13 using exact ancestry and ownership. Cache memory retired after each prediction, and actual-worker and boundary checks passed.

Recorded measurement:

    Native2 bill-0.1158296%;all16 bytes exact;13hits/3paid copies/network;337MB cache;319,701,476F and296 requests saved;peak5.997GB,48,637handles,zero swap.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_uvp_base_bcache_v1/candidate.py](code/hill_climb_uvp_base_bcache_v1/candidate.py)
- [code/hill_climb_uvp_base_bcache_v1/build.py](code/hill_climb_uvp_base_bcache_v1/build.py)
- [code/hill_climb_star_projection32_v1/candidate.py](code/hill_climb_star_projection32_v1/candidate.py)
- [code/hill_climb_uvp_base_bcache_v1/fragment.py](code/hill_climb_uvp_base_bcache_v1/fragment.py)

## Evidence

- [HILL_CLIMB_before_exact_bundle_v1.md#L13](HILL_CLIMB_before_exact_bundle_v1.md#L13)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_before_exact_bundle_v1.md](HILL_CLIMB_before_exact_bundle_v1.md) · record
- [RESULT.md](RESULT.md) · record
- [retained_component.json](retained_component.json) · result
- [build.json](build.json) · result
- [PROTOCOL.md](PROTOCOL.md) · record

</details>

## Related work

- shipped in: [Submission #332838: precision changes with symmetric 128-row diagonal covariance blocks](../../../submissions/332838/README.md)

[Research graph](../../../graph/index.html#node=2026-09-28-immutable-common-difference-base-factor-cache-all-outputs-exact) · [Copy provenance](../../../PROVENANCE.json)

