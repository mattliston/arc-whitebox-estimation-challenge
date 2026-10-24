# Five byte-identical request reductions save 390 cold and 386 warm calls, funding a 0.47% bill reduction

2026-10-16T16:05:51Z · experiment · adopted · measured

Five line-preserving editors removed client requests from the estimator submitted as #336412 (S2tr) without changing any prediction bit: strip-path temporaries kept via the output-view cache, six diagonal re-reads in the dense core reused, three fill_diagonal sites fed float64 directly and the pair map reusing core diagonals, a changed-row count diagnostic and scalar fetches folded, and a process-lifetime index array for the inverse-map scatters. Together with the release RQX2 editor (-97) the freed budget let _GSAFE5_LIMIT go to 0 (all five full covariance sandwiches become two-strip), which S3 carried; the editors were later composed into the fired runtime lineage (audited RQS2/RQX2/MS in the v11 release source).

Recorded measurement:

    Pieces (cold / warm): strips -11 / -4; fhmdiag -61 / -61; pairdiag -85 / -85; misc -118 / -120; reuse -115 / -116; cumulative -390 / -386, flops lower, residual instructions x0.990 / x0.989. RQX2 -97 / -97, flops -59K. One _GSAFE5_LIMIT step = +94 requests and -0.093% flops (linear 5..2); limit 0 = -0.47% bill; on S1b slow 42,394 / 40,775 (margins 306 / 425), instruction-weighted median .3153. The 159 _l3o_last parent-path calls are real product outputs and cannot be kept.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_rqs2_v1/editor_rqs2_strips_v3.py](code/opus_rqs2_v1/editor_rqs2_strips_v3.py)
- [code/opus_rqs2_v1/editor_rqs2_fhmdiag_v2.py](code/opus_rqs2_v1/editor_rqs2_fhmdiag_v2.py)
- [code/opus_rqs2_v1/editor_rqs2_pairdiag_v1.py](code/opus_rqs2_v1/editor_rqs2_pairdiag_v1.py)
- [code/opus_rqs2_v1/editor_rqs2_misc_v2.py](code/opus_rqs2_v1/editor_rqs2_misc_v2.py)
- [code/opus_rqs2_v1/editor_rqs2_reuse_v1.py](code/opus_rqs2_v1/editor_rqs2_reuse_v1.py)
- [code/opus_rqs2_v1/cand_s2tr_rqs2e_v1.py](code/opus_rqs2_v1/cand_s2tr_rqs2e_v1.py)
- [code/opus_rqs2_v1/build_g5size_v2.py](code/opus_rqs2_v1/build_g5size_v2.py)
- [code/opus_passrel_v1/rqx2_v1/editor_rqx2_v1.py](code/opus_passrel_v1/rqx2_v1/editor_rqx2_v1.py)
- [code/opus_passrel_v1/s1_v1/cand_s1b_stack_preview_v1.py](code/opus_passrel_v1/s1_v1/cand_s1b_stack_preview_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4272](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4272)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4282](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4282)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4293](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4293)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4299](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4299)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4309](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4309)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4315](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4315)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4317](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4317)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4657](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4657)

<details>
<summary>All result files and records (2)</summary>

- [frag_gsafe5_limit0_v1.py.txt](frag_gsafe5_limit0_v1.py.txt) · code
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Static low-residual stack projects 1.4747e-9 on a boundary pair but only ties the baseline on eight networks](../../submission-process/static-low-residual-stack-projects-1-4747e-9-boundary/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-five-byte-identical-request-reductions-save-390-cold-386) · [Copy provenance](../../../PROVENANCE.json)

