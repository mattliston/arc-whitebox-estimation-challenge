# Response floors, larger selection counts and plain products fail to improve cost and error together

2026-10-01T21:22Z · experiment · rejected · measured

Three kernel-side levers were priced on the screen networks against the parent: lowering the response floor to 0.0075 trades bill for error about one to one, selection counts rounded to 64 with real units raise both bill and error, and plain matmul everywhere costs +90.2% bill for +0.08% MSE while running 13 s per network instead of 31. The 3.5% bill target was judged unreachable by kernel work; the remaining named levers sum to about -2.2% with the half-zero strips.

Recorded measurement:

    response floor 0.0075: bill +0.705% / MSE -0.805% / product -0.09%; selection counts to 64: bill +0.110% / MSE +0.644%; plain matmul everywhere: bill +90.2% / MSE +0.08% / 13 s vs 31 s per network; remaining levers: row cuts rounded to 32 about -0.35%, 7-level strips about -0.25%, cheaper tanh in the edge about -0.3%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a01_response_floor/build.py](code/opus_reexam_20261001_v1/engineer/a01_response_floor/build.py)
- [code/opus_reexam_20261001_v1/engineer/a01_response_floor/candidate_f0p0075.py](code/opus_reexam_20261001_v1/engineer/a01_response_floor/candidate_f0p0075.py)
- [code/opus_reexam_20261001_v1/engineer/a02_select64/build.py](code/opus_reexam_20261001_v1/engineer/a02_select64/build.py)
- [code/opus_reexam_20261001_v1/engineer/a02_select64/candidate_lrf.py](code/opus_reexam_20261001_v1/engineer/a02_select64/candidate_lrf.py)
- [code/opus_reexam_20261001_v1/engineer/buildlib.py](code/opus_reexam_20261001_v1/engineer/buildlib.py)
- [code/opus_reexam_20261001_v1/engineer/d01_plain_matmul/build.py](code/opus_reexam_20261001_v1/engineer/d01_plain_matmul/build.py)
- [code/opus_reexam_20261001_v1/engineer/d01_plain_matmul/candidate_plain.py](code/opus_reexam_20261001_v1/engineer/d01_plain_matmul/candidate_plain.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L37](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L37)

<details>
<summary>All result files and records (2)</summary>

- [RESULT.md](../half-zero-strip-kernel-odd-32-dimensions-lowers-charged/RESULT.md) · result
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-response-floors-larger-selection-counts-plain-products-fail-improve) · [Copy provenance](../../../PROVENANCE.json)

