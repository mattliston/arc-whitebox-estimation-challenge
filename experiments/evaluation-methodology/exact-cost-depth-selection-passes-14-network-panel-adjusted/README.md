# Exact-cost depth selection passes 14-network panel: adjusted ratio .954075238, score 3.9806136484e-9

2026-09-19 · experiment · adopted · measured

The Cost150 candidate (exact kernel-cost schedule plus scalar final readout on the wall-repaired card) was scored on the fixed 8 public + 6 virgin panel against the I95 bank with independent 4M truths and noise correction. Both groups improved and the whole panel fell under the local 4e-9 threshold; the official projection by bank ratio was 3.9960089480e-9.

Recorded measurement:

    public8 NC adjusted 4.3995605705e-9 vs 4.6220625306e-9 (ratio .951860894, 8/8); virgin6 3.4220177522e-9 vs 3.5724345057e-9 (.957895168, 5/6); all14 3.9806136484e-9 vs 4.1722219485e-9 (.954075238, 13/14); average cold 416.157BF vs 442.128BF; projection 3.9960089480e-9

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cost150_panel.py](code/cost150_panel.py)
- [code/score_cost150_panel.py](code/score_cost150_panel.py)
- [code/build_cost_rule.py](code/build_cost_rule.py)
- [code/cost_rule150_scalar.py](code/cost_rule150_scalar.py)
- [code/native_screen.py](code/native_screen.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/scalar_distill_I95.py](code/scalar_distill_I95.py)

## Evidence

- [RECEIPTS.md#L122](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L122)
- [RECEIPTS.md#L126](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L126)

<details>
<summary>All result files and records (3)</summary>

- [cost150_panel_score.json](cost150_panel_score.json) · result
- [cost150_package.json](cost150_package.json) · record
- [RECEIPTS.md](../../birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

## Related work

- shipped in: [Submission #331584: exact-cost Winograd depth selection and a scalar final readout on the wall-repaired estimator](../../../submissions/331584/README.md)

[Research graph](../../../graph/index.html#node=2026-09-19-exact-cost-depth-selection-passes-14-network-panel-adjusted) · [Copy provenance](../../../PROVENANCE.json)

