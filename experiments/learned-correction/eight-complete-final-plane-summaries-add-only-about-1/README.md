# Eight complete final-plane summaries add only about 1% beyond the affine oracle and miss both gates

2026-09-21 · experiment · rejected · measured

Full covariance and repeated-third-plane row summaries were screened with separate network-specific coefficients and zero added inference cost.

Recorded measurement:

    | Existing 1B network | Ideal / parent | Ideal / affine control | Served / parent | Served / affine control |
    | 35 | 0.880978601109 | 0.992102057743 | 0.878351394848 | 0.991217255851 |
    | 100 | 0.883441280077 | 0.987926830649 | 0.880118318074 | 0.986550088563 |
    Each network required both ideal and served ratios <=0.85 against the
    unchanged parent and <=0.90 against the matching affine oracle. All eight
    comparisons fail. Ideal extra improvement beyond affine is 0.790% / 1.207%;
    served extra improvement is 0.878% / 1.345%. The roughly 12% reduction against
    the parent includes the privileged refit of its existing 65-column readout.
    asks a different question from the previous 65 hidden-direction queries,
    while reusing their authenticated exact-331684 captures. No new physical
    Both controls have full rank (65 and 73). Independent QR solves using
    all clamped FP32 prediction bytes and every gate decision. Maximum coefficient
    relative difference is 2.0828350495e-14;
    1.38051725916e-17. Privileged coefficients
    unclipped FP64; served predictions are nonnegative FP32.
    6.22282396967e-16 against 2e-10. Wide independent 8-by-8 plane
    and covariance slices agree within 1.93450976983e-15
    and 4.13865854891e-16, respectively.
    Formal FP64 diagonals differ from native k3 by at most
    7.02414581569e-07; covariance diagonals differ
    from the original sigma squared by at most 2.63095526419e-06.
    Those checks meet their fixed 1e-3 tolerances; they do not assert exact equality
    to the native packed arithmetic. Inputs are unchanged and 35/100/35 feature
    413448696909 F for 35 and 416264792412 F for 100. The old warm capture's
    100 bill is not used as its cold cost. Research FP64 feature contraction

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260919_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/terminal_full_plane_information_v1/audit.py](code/terminal_full_plane_information_v1/audit.py)
- [code/terminal_full_plane_information_v1/common.py](code/terminal_full_plane_information_v1/common.py)
- [code/terminal_full_plane_information_v1/compute.py](code/terminal_full_plane_information_v1/compute.py)
- [code/terminal_full_plane_information_v1/feature.py](code/terminal_full_plane_information_v1/feature.py)
- [code/terminal_full_plane_information_v1/finalize.py](code/terminal_full_plane_information_v1/finalize.py)
- [code/terminal_full_plane_information_v1/oracle.py](code/terminal_full_plane_information_v1/oracle.py)
- [code/terminal_full_plane_information_v1/prepare.py](code/terminal_full_plane_information_v1/prepare.py)
- [code/terminal_full_plane_information_v1/run.py](code/terminal_full_plane_information_v1/run.py)
- [code/terminal_full_plane_information_v1/unit.py](code/terminal_full_plane_information_v1/unit.py)
- [code/full_lower_covariance_edge_packed_v1/candidate.py](code/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/run_guarded.py](code/run_guarded.py)
- [code/terminal_cubic_hidden_queries_v1/capture.py](code/terminal_cubic_hidden_queries_v1/capture.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/terminal_cubic_hidden_queries_v1/common.py](code/terminal_cubic_hidden_queries_v1/common.py)

</details>

## Evidence

- [RESULT.md#L3](RESULT.md#L3)
- [RESULT.md#L9](RESULT.md#L9)
- [RESULT.md#L11](RESULT.md#L11)
- [RESULT.md#L12](RESULT.md#L12)
- [RESULT.md#L14](RESULT.md#L14)
- [REEXAMINATION_INDEX_20260919.md#L701](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L701)
- [REEXAMINATION_INDEX_20260919.md#L704](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L704)

<details>
<summary>All result files and records (6)</summary>

- [RESULT.md](RESULT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [closeout.json](closeout.json) · result
- [result.json](result.json) · result
- [summary.json](summary.json) · result
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-21-eight-complete-final-plane-summaries-add-only-about-1) · [Copy provenance](../../../PROVENANCE.json)

