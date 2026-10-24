# Rank256 full signed-tensor sketch gives 43.0325948333x query error versus rank384 parent

2026-09-20 · experiment · rejected · measured

The construction used 256 probes and optimistic FP64 algebra, then evaluated 512 independent verification probes and the full repeated-index plane.

Recorded measurement:

    The fixed rank256 /256-construction-probe QR basis failed at the declared
    current Lazy rank35 layer8 join. Its512 independent verification probes give
    residual squared-energy ratio **43.0325948333** versus
    the actual rank384 parent, with0/512 query wins. The complete
    next repeated-index plane gives ratio **39.5698945013**.
    Both ratios had to be<=.75. This decisive failure stops the construction.
    | Same current join tensor | Existing rank384 | FP64 sketch rank256 |
    | Relative query residual squared energy | 0.00116903789082 | 0.0503067339005 |
    | Relative full-plane residual squared energy | 0.000403428661158 | 0.0159636295608 |
    Marginal diagonal residual-energy ratio is32.6801257907,
    18.5873396756, with empirical standard error
    0.209101820371 over these512 fixed random probes.
    FP64 algebra; the comparison uses the parent's actual rounded output factors.
    The capture has896 active rows,6144 prefix columns and1984 entering columns,
    8128 total. All16 Lazy output bytes and its exact398080917697F cold bill match
    implementation claim. The original cache retires; the54,920,372-byte packet
    projection with maximum scaled error7.673381875059282e-14. The v1 sum-individual
    cube diagnostic was not a complete-pair comparison; an additive v2 exact
    while their summed individual pair Grams equal diag(2,2,0). Both records and
    orthogonality error is1.6967695501e-15.
    All four guarded stages exit0 and retire. Peak RSS including self-reports is
    5551390720 bytes, zero added swap. Analysis takes3.475765s and peaks at
    738066432 bytes. No rank100, other join, extra probes, seed/rank/
    follows this failure. Earlier pair384 and whole-pair512 negatives stay closed.
    Future qualified Phase2 uploads have the user's explicit preauthorization.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/full_tensor_sketch_information_v1/algebra.py](code/full_tensor_sketch_information_v1/algebra.py)
- [code/full_tensor_sketch_information_v1/capture.py](code/full_tensor_sketch_information_v1/capture.py)
- [code/full_tensor_sketch_information_v1/finalize.py](code/full_tensor_sketch_information_v1/finalize.py)
- [code/full_tensor_sketch_information_v1/information.py](code/full_tensor_sketch_information_v1/information.py)
- [code/full_tensor_sketch_information_v1/pair_gram_certificate_v2.py](code/full_tensor_sketch_information_v1/pair_gram_certificate_v2.py)
- [code/full_tensor_sketch_information_v1/unit.py](code/full_tensor_sketch_information_v1/unit.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L4](NEGATIVE_RESULT.md#L4)
- [NEGATIVE_RESULT.md#L5](NEGATIVE_RESULT.md#L5)
- [NEGATIVE_RESULT.md#L6](NEGATIVE_RESULT.md#L6)

<details>
<summary>All result files and records (3)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-rank256-full-signed-tensor-sketch-gives-43-0325948333x-query) · [Copy provenance](../../../PROVENANCE.json)

