# Thin star products routed to depth three: eight training scores improve by a pooled 0.573464%

2026-09-26T23:11Z · experiment · adopted · measured

Six products across four fixed shapes used the existing depth-three kernel with ranks, learned weights and setup unchanged. The component passed training expansion but remained below the submission threshold without held-out or worker qualification.

Recorded measurement:

    TRAIN8 adjusted-0.573464%, paired training-bootstrap95% interval[-0.733173%,-0.413888%],8/8 wins; raw+0.051701%; bill-0.625005%; fresh6 CPU+2.53412%;87 routed products/network,zero fallbacks.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate_thin_star_route_v1.py](code/candidate_thin_star_route_v1.py)

## Evidence

- [RESULTS.md#L752](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md#L752)

<details>
<summary>All result files and records (2)</summary>

- [RESULTS.md](../../third-order-planes/all-old-tensor-reconstruction-replaced-trace-adjusted-error-rises/RESULTS.md) · record
- [thin_star_route_training_component_v1.json](thin_star_route_training_component_v1.json) · result

</details>

## Related work

- shipped in: [Submission #332738: rank-32 coordinate-star projection](../../../submissions/332738/README.md)

[Research graph](../../../graph/index.html#node=2026-09-26-thin-star-products-routed-depth-three-eight-training-scores) · [Copy provenance](../../../PROVENANCE.json)

