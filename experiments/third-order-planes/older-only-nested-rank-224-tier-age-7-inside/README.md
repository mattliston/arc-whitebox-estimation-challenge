# Older-only nested rank-224 tier at age 7 inside the rank-384 shared tier: pooled adjusted ratio 1.011419853 (rejected)

2026-09-19 · experiment · rejected · measured

At layers 8..14 exactly one oldest cohort was moved into a rank-224 map inside the rank-384 coordinates by two weighted-Gram rangefinder QR passes, with the core updated by subtracting the mover's original core and adding its projected core; the published F73 implementation had reported an 8-network gain of about 3.2% from such a second tier. Built on the Cost150 source, it cut the cold bill about 1.9% but raised raw error more, failing on net 100 and pooled.

Recorded measurement:

    own cold35 F 408.171BF vs 415.942BF, cold100 411.352BF vs 418.740BF; raw ratios 1.017974714 / 1.042294577; adjusted ratios .998956748 / 1.023904445, pooled 1.011419853; two-move synthetic check maximum relative 1.30e-7; candidate SHA eb16f16f1fadfb4455da0628d4220d632ad1a6a43f1f4cde0b36e028b239fe53

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/nested224_cost150.py](code/nested224_cost150.py)
- [code/build_nested224.py](code/build_nested224.py)
- [code/nested224_certificate.py](code/nested224_certificate.py)
- [code/score_nested224_dev.py](code/score_nested224_dev.py)
- [code/native_screen.py](code/native_screen.py)
- [code/cost_rule150_scalar.py](code/cost_rule150_scalar.py)
- [code/nested224_fragment.py](code/nested224_fragment.py)

## Evidence

- [RECEIPTS.md#L188](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L188)
- [RECEIPTS.md#L197](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L197)
- [RECEIPTS.md#L201](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L201)
- [NESTED224_PROTOCOL.md#L9](NESTED224_PROTOCOL.md#L9)

<details>
<summary>All result files and records (5)</summary>

- [NESTED224_PROTOCOL.md](NESTED224_PROTOCOL.md) · record
- [nested224_dev_score.json](nested224_dev_score.json) · result
- [nested224_certificate.json](nested224_certificate.json) · result
- [nested224_build.json](nested224_build.json) · config
- [RECEIPTS.md](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-19-older-only-nested-rank-224-tier-age-7-inside) · [Copy provenance](../../../PROVENANCE.json)

