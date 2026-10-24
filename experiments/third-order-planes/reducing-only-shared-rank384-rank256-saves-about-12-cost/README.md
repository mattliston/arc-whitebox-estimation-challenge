# Reducing only shared rank384 to rank256 saves about 12% cost but worsens adjusted error 18.657200%

2026-09-20 · experiment · rejected · measured

Four width literals changed while age, metric, selectors, learned values and delayed-rotation schedule remained fixed.

Recorded measurement:

    # Plain shared-rank256 current-Lazy endpoint: objective failure
    Candidate588b211fb06b32e6d878532c82039bc7ca1560ad9ffec6602f2e4a9d3dcf0c36
    changes only four384->256 literal nodes: LAST shared push rank/minimum guard,
    restores exact parentb7c21ac7. Age4, original lambda-squared Gram/Omega rule,
    unmeasured rank-only endpoint identified by RANK256_RECEIPT_SCOPE; it does not
    rewrite or rescue the old joint-change PAIR256 or spectral-query experiments.
    |35|1.87060284900779e-08|2.49139029735438e-08|350641667199|0.880830131792179|1.17314674524762|
    |100|1.84245341752749e-08|2.50715326193113e-08|353560758058|0.881932840134297|1.20010675760485|
    The fixed <=.95 adjusted gate FAILS on both networks. Raw error rises
    33.186491%/36.076887%, overwhelming approximately12% cost savings. Pooled
    adjusted ratio1.186572000836679,18.657200% worse. These are ordinary1B
    noise-corrected14 panel or official score. Every parent raw/adjusted value is
    First5 parent output layers are byte exact; every candidate row is finite,
    FP32 and nonnegative; existing finite/layout/overflow guards pass. Native
    35/100/35 and independent cold100 preserve own repeat and warm/cold output
    bytes exactly. Both W5 contexts finish and numeric caches retire on each call.
    The final candidate-minus-parent prediction MSE is6.122322663142234e-9 and
    6.389062331709114e-9. Subtracting that and the parent MSE from the candidate
    refit or kernel rescue. No independent14, confirmation, client, contract,
    rank256 bound remains scoped to its expected tensor-query criterion; this
    All3 guards completed and retired;4 full predictions,
    1377993816416F,109.854097s summed guarded wall,
    peak6702006272B including available worker self-reports,
    zero added swap. Tool sessions89966/7476 TERMINAL; no owned numeric/upload
    streamed scratch already failed timing; fixed GaussianR1, fourth emission,

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/rank256_objective_v1/build.py](code/rank256_objective_v1/build.py)
- [code/rank256_objective_v1/candidate.py](code/rank256_objective_v1/candidate.py)
- [code/rank256_objective_v1/score_dev.py](code/rank256_objective_v1/score_dev.py)
- [code/lazy_rotation_objective_v1/candidate.py](code/lazy_rotation_objective_v1/candidate.py)

## Evidence

- [NEGATIVE_RESULT.md#L3](NEGATIVE_RESULT.md#L3)
- [NEGATIVE_RESULT.md#L1](NEGATIVE_RESULT.md#L1)
- [NEGATIVE_RESULT.md#L8](NEGATIVE_RESULT.md#L8)
- [NEGATIVE_RESULT.md#L9](NEGATIVE_RESULT.md#L9)
- [NEGATIVE_RESULT.md#L11](NEGATIVE_RESULT.md#L11)

<details>
<summary>All result files and records (4)</summary>

- [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [dev_score.json](dev_score.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-20-reducing-only-shared-rank384-rank256-saves-about-12-cost) · [Copy provenance](../../../PROVENANCE.json)

