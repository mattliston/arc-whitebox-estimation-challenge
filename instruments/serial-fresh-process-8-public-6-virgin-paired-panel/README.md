# Serial fresh-process 8 public + 6 virgin paired panel: sealed outputs, noise-corrected scoring, guarded supervisor

2026-09-19 · tool · adopted · derived

The 09-19 continuation evaluates every candidate with paired_panel.py (28 serial fresh processes over 14 nets, outputs and complete cold bills sealed before the independent 4M truths are opened), score_paired_panel.py (ordinary and noise-corrected MSE reported separately), native_screen.py (native outputs and bills without truth access), boundary_client.py and boundary_driver.py (NumPy-free production-client qualification against a local flopscope server) and run_guarded.py (a serial process-group supervisor bounding RAM, swap and elapsed time). Noise-corrected adjusted means subtract the Monte Carlo reference variance; ordinary 4M scores include about 1.83e-8 of reference noise.

Recorded measurement:

    ordinary 4M adjusted 7.84660e-9 / 7.69568e-9 vs noise-corrected 4.1722219485e-9 / 4.1183865223e-9 for the same two arms

## Programs

- [code/codex_resume_20260919_v1/paired_panel.py](code/codex_resume_20260919_v1/paired_panel.py)
- [code/codex_resume_20260919_v1/score_paired_panel.py](code/codex_resume_20260919_v1/score_paired_panel.py)
- [code/codex_resume_20260919_v1/native_screen.py](code/codex_resume_20260919_v1/native_screen.py)
- [code/codex_resume_20260919_v1/boundary_client.py](code/codex_resume_20260919_v1/boundary_client.py)
- [code/codex_resume_20260919_v1/boundary_driver.py](code/codex_resume_20260919_v1/boundary_driver.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)
- [code/codex_resume_20260919_v1/scalar_distill_I95.py](code/codex_resume_20260919_v1/scalar_distill_I95.py)
- [code/fable_wall_repair_v1/cand_c2_wallrepair_I95.py](code/fable_wall_repair_v1/cand_c2_wallrepair_I95.py)

## Evidence

- [RECEIPTS.md#L56](../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L56)
- [RECEIPTS.md#L73](../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md#L73)
- [paired_panel.py#L1](code/codex_resume_20260919_v1/paired_panel.py#L1)
- [run_guarded.py#L1](code/codex_resume_20260919_v1/run_guarded.py#L1)

<details>
<summary>All result files and records (2)</summary>

- [paired14_score.json](../../experiments/learned-correction/scalar-final-readout-distilled-head-s-predictions-14-net/paired14_score.json) · result
- [RECEIPTS.md](../../experiments/birth-terms/birth-only-common-base-rank-128-projection-newborn-signed/RECEIPTS.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-19-serial-fresh-process-8-public-6-virgin-paired-panel) · [Copy provenance](../../PROVENANCE.json)

