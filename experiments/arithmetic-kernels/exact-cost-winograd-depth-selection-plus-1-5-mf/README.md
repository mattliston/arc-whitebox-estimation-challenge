# Exact-cost Winograd depth selection plus 1.5 MF per request improves adjusted error: ratio .957179078

2026-09-19 · experiment · adopted · measured

The shape thresholds of the wall-repair size rule were replaced by an exact kernel cost formula (reproducing every saved bill of the 178-shape benchmark at all supported depths) plus a fixed penalty of 1.5 MFLOP per remote request, chosen from shape and cost evidence only; shapes select square depths 0/3/4/5, read depths 0/4 and push depths 0/4/6 with the original square maxima retained. On net 35 the schedule cut the cold bill from 442.087B to 415.942B FLOPs at unchanged physics, but the production client needed more requests and wall time than the bank.

Recorded measurement:

    native35 cold 415.942BF, raw 1.869085768e-8, adjusted 3.535346098e-9, ratio .957179078 to I95; client cold 51,391 requests / 36.64 s vs bank 45,117 / 30.14 s; next100 43,697 / 33.92 s vs 39,004 / 27.98 s; cold residual .2944 s, later .145 s

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/cost_rule_plan.py](code/codex_resume_20260919_v1/cost_rule_plan.py)
- [code/codex_resume_20260919_v1/build_cost_rule.py](code/codex_resume_20260919_v1/build_cost_rule.py)
- [code/codex_resume_20260919_v1/cost_rule150_scalar.py](code/codex_resume_20260919_v1/cost_rule150_scalar.py)
- [code/codex_resume_20260919_v1/scalar_distill_I95.py](code/codex_resume_20260919_v1/scalar_distill_I95.py)
- [code/fable_wall_repair_v1/cand_c2_wallrepair_I95.py](code/fable_wall_repair_v1/cand_c2_wallrepair_I95.py)

## Evidence

- [RECEIPTS.md#L103](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L103)
- [RECEIPTS.md#L111](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L111)
- [RECEIPTS.md#L114](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L114)

<details>
<summary>All result files and records (4)</summary>

- [cost_rule150_35_score.json](cost_rule150_35_score.json) · result
- [cost_rule_plan.json](cost_rule_plan.json) · result
- [cost150_boundary_watch.json](cost150_boundary_watch.json) · log
- [RECEIPTS.md](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

## Related work

- shipped in: [Submission #331584: exact-cost Winograd depth selection and a scalar final readout on the wall-repaired estimator](../../../submissions/331584/README.md)
- ← enables: [Scalar final readout distilled from the head's predictions: 14-net noise-corrected adjusted ratio .9870967012 (held)](../../learned-correction/scalar-final-readout-distilled-head-s-predictions-14-net/README.md)

[Research graph](../../../graph/index.html#node=2026-09-19-exact-cost-winograd-depth-selection-plus-1-5-mf) · [Copy provenance](../../../PROVENANCE.json)

