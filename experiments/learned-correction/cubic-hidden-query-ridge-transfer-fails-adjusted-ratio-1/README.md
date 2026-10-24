# Cubic hidden-query ridge transfer fails: adjusted ratio 1.030768487997, 1/20 networks improve

2026-09-21 · experiment · rejected · measured

A fixed five-fold cross-network regression tested the same 65 cubic features without deploying privileged oracle coefficients.

Recorded measurement:

    The unchanged65 cubic queries from terminal_cubic_hidden_queries_v1 were
    oracle still fails its .80 opportunity gates. This study explicitly revised
    16 fit and4 held per model. Each fold fitted the65-column affine control and
    the130-column extended design with fit-only weighted RMS normalization and
    ridge1e-3. Every model's served networks are absent from its training labels.
    - Pooled adjusted extended/parent: 1.030768487997 (required<=.99).
    - Pooled adjusted extended/affine: 1.040783953271 (required<=.99).
    - Individual parent wins: 1/20 (required>=12).
    - Parent-improving folds: 0/5 (required>=4).
    - Worst individual adjusted ratio: 1.078915190367 (required<=1.20).
    for each network, plus a provisional2MF readout allowance. The affine control
    pays parent+2MF. The isolated wide extended readout costs1206272 F and the
    affine readout673792 F, within that allowance. Full native candidate bills
    All23 cold parent calls reproduce all16 bank output rows and complete bills
    exactly, and retire original caches. Qualification35/100 reproduces the
    earlier feature bytes/bills; repeated35 also matches all compact fields and
    bills. All wide queries pass independent FP64 source reconstruction at2e-10
    relative and formal/native third diagonal comparison at1e-3. Only compact
    fixtures pass independent augmented least squares; all10 wide fits pass
    solve roundoff by the1e-12 scale floor. The preserved diagnostic establishes
    scaled agreement<=6.27e-14 and prediction agreement<=1.28e-15. Additive v2
    checks both in meaningful coordinates at the original2e-10 tolerance.
    Resources:23 full parent calls, 9527648743374 F, plus
    330661391954 F of measured helpers;10 CV fits. All30 guarded
    groups, including the failed fixture, are retired. Controllers65020/88499

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_cubic_transfer_v1/capture.py](code/terminal_cubic_transfer_v1/capture.py)
- [code/terminal_cubic_transfer_v1/capture_v2.py](code/terminal_cubic_transfer_v1/capture_v2.py)
- [code/terminal_cubic_transfer_v1/common.py](code/terminal_cubic_transfer_v1/common.py)
- [code/terminal_cubic_transfer_v1/finalize.py](code/terminal_cubic_transfer_v1/finalize.py)
- [code/terminal_cubic_transfer_v1/fit.py](code/terminal_cubic_transfer_v1/fit.py)
- [code/terminal_cubic_transfer_v1/helper.py](code/terminal_cubic_transfer_v1/helper.py)
- [code/terminal_cubic_transfer_v1/observer_source.py](code/terminal_cubic_transfer_v1/observer_source.py)
- [code/terminal_cubic_transfer_v1/preflight.py](code/terminal_cubic_transfer_v1/preflight.py)
- [code/terminal_cubic_transfer_v1/preflight_v2.py](code/terminal_cubic_transfer_v1/preflight_v2.py)
- [code/terminal_cubic_transfer_v1/prepare.py](code/terminal_cubic_transfer_v1/prepare.py)
- [code/terminal_cubic_transfer_v1/prepare_repair.py](code/terminal_cubic_transfer_v1/prepare_repair.py)
- [code/terminal_cubic_transfer_v1/qualify.py](code/terminal_cubic_transfer_v1/qualify.py)

<details>
<summary>Additional programs and dependencies (15)</summary>

- [code/terminal_cubic_transfer_v1/qualify_v2.py](code/terminal_cubic_transfer_v1/qualify_v2.py)
- [code/terminal_cubic_transfer_v1/run.py](code/terminal_cubic_transfer_v1/run.py)
- [code/terminal_cubic_transfer_v1/run_v2.py](code/terminal_cubic_transfer_v1/run_v2.py)
- [code/terminal_cubic_transfer_v1/score.py](code/terminal_cubic_transfer_v1/score.py)
- [code/terminal_cubic_transfer_v1/solver.py](code/terminal_cubic_transfer_v1/solver.py)
- [code/terminal_cubic_transfer_v1/solver_diagnostic.py](code/terminal_cubic_transfer_v1/solver_diagnostic.py)
- [code/terminal_cubic_transfer_v1/solver_unit.py](code/terminal_cubic_transfer_v1/solver_unit.py)
- [code/terminal_cubic_transfer_v1/solver_unit_v2.py](code/terminal_cubic_transfer_v1/solver_unit_v2.py)
- [code/terminal_cubic_transfer_v1/unit.py](code/terminal_cubic_transfer_v1/unit.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/terminal_cubic_hidden_queries_v1/capture.py](code/terminal_cubic_hidden_queries_v1/capture.py)
- [code/terminal_cubic_hidden_queries_v1/common.py](code/terminal_cubic_hidden_queries_v1/common.py)
- [code/terminal_cubic_hidden_queries_v1/helper.py](code/terminal_cubic_hidden_queries_v1/helper.py)
- [code/terminal_cubic_hidden_queries_v1/unit.py](code/terminal_cubic_hidden_queries_v1/unit.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L5](RESULT.md#L5)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L12](RESULT.md#L12)
- [REEXAMINATION_INDEX_20260919.md#L654](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L654)
- [REEXAMINATION_INDEX_20260919.md#L657](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L657)

<details>
<summary>All result files and records (5)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-cubic-hidden-query-ridge-transfer-fails-adjusted-ratio-1) · [Copy provenance](../../../PROVENANCE.json)

