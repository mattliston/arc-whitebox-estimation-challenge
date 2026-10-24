# Exact-cost depth selector with a 750k instead of 1.5M FLOP request penalty: 14-net adjusted ratio .9906359025, held

2026-09-19 · experiment · inconclusive · measured

Cost75 lowers the shape-only per-request penalty of the exact kernel-cost selector from 1,500,000 to 750,000 estimated FLOPs with physics and payloads unchanged; the archived census predicted 4.205BF saved for 3,747 more warm kernel requests. It passed development, the 14-net panel and production-client parity with more requests and slightly longer walls; its upload was first blocked by the automatic review and, once approved, it was superseded by stronger candidates and never fired.

Recorded measurement:

    dev adjusted ratios 35 .9927268204, 100 .9897427038, pooled .9912360472; public8 NC 4.3533863365e-9 vs 4.3995605705e-9 (.9895048078, 8/8); fresh6 3.3966087377e-9 vs 3.4220177522e-9 (.9925748443, 5/6); all14 3.9433387942e-9 vs 3.9806136484e-9 (.9906359025, 13/14); mean cold bill 412.655BF vs 416.157BF; client cold 37.304 s / 56,288 requests vs 36.637 s / 51,391; projection 3.9779338937e-9; candidate SHA 5f66f94afdd2909584527a299c3989377d374289b8f5e1f379047b4176c7521c

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cost_rule75_scalar.py](code/cost_rule75_scalar.py)
- [code/build_cost75.py](code/build_cost75.py)
- [code/cost75_panel.py](code/cost75_panel.py)
- [code/score_cost75_panel.py](code/score_cost75_panel.py)
- [code/cost_rule150_scalar.py](code/cost_rule150_scalar.py)
- [code/native_screen.py](code/native_screen.py)
- [code/run_guarded.py](code/run_guarded.py)

## Evidence

- [RECEIPTS.md#L233](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L233)
- [RECEIPTS.md#L244](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L244)
- [RECEIPTS.md#L249](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L249)
- [RECEIPTS.md#L254](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L254)
- [COST75_PROTOCOL.md#L6](COST75_PROTOCOL.md#L6)

<details>
<summary>All result files and records (6)</summary>

- [COST75_PROTOCOL.md](COST75_PROTOCOL.md) · record
- [cost75_dev_score.json](cost75_dev_score.json) · result
- [cost75_panel_score.json](cost75_panel_score.json) · result
- [cost75_boundary_comparison.json](cost75_boundary_comparison.json) · result
- [cost75_package.json](cost75_package.json) · record
- [RECEIPTS.md](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

## Related work

- supersedes: [Submission #331584: exact-cost Winograd depth selection and a scalar final readout on the wall-repaired estimator](../../../submissions/331584/README.md)

[Research graph](../../../graph/index.html#node=2026-09-19-exact-cost-depth-selector-750k-instead-1-5m-flop) · [Copy provenance](../../../PROVENANCE.json)

