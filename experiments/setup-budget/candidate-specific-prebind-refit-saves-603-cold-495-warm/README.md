# Candidate-specific prebind refit saves 603 cold and 495 warm requests under slow setup, with identical bytes

2026-10-16T17:00:42Z · experiment · adopted · measured

S3 prebind list was refitted on its own 20-network key census with the targeted prebind tool (100 listed keys dead for S3, 26 swapped for keys used by at least 2 networks, usage reorder) plus a refit of the k-split accumulation key list. Predictions stayed byte-identical; the slow first-network margin went from 310 to 913 requests. The 8-network session showed the bill gain is mostly on the cold network. The refit procedure (census, prebind_targeted_v2, k-split refit) was reused for later fired files, including the v61j list of #336867.

Recorded measurement:

    Requests normal first 41,929 -> 41,503 (-426) / warm 40,634 -> 40,210 (-424), slow 42,390 -> 41,787 (-603) / 40,772 -> 40,277 (-495); simulated slow first-net lazy binds 661 -> 457; flops normal -0.096% / -0.076%, slow -0.150% / -0.155%; keys bound 587 -> 589 / 403 -> 408. 8-net session: byte-identical on all 8, requests -426 cold and -272..-424 later (mean -342), flops cold -0.096%, nets 2-8 mean -0.024%, residual instructions cold -1.04%, nets 2-8 mean -0.08%. S3pb prereg 1.4736e-9 boundary (~1.4757e-9 8-net).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_s3pb_v1/refit_s3pb_v1.sh](code/opus_s3pb_v1/refit_s3pb_v1.sh)
- [code/opus_s3pb_v1/cand_s3pb_v1.py](code/opus_s3pb_v1/cand_s3pb_v1.py)
- [code/opus_passrel_v1/stack_v1/cand_s3pb.py](code/opus_passrel_v1/stack_v1/cand_s3pb.py)
- [code/opus_passrel_v1/stack_v1/cand_s3.py](code/opus_passrel_v1/stack_v1/cand_s3.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4356](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4356)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4365](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4365)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4367](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4367)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4371](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4371)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4372](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4372)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4374](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4374)

<details>
<summary>All result files and records (4)</summary>

- [fragment_prebind_s3_v1.py.txt](fragment_prebind_s3_v1.py.txt) · code
- [fragment_ks_acc_s3_v1.py.txt](fragment_ks_acc_s3_v1.py.txt) · code
- [STACK_s3pb.json](STACK_s3pb.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Static low-residual stack projects 1.4747e-9 on a boundary pair but only ties the baseline on eight networks](../../submission-process/static-low-residual-stack-projects-1-4747e-9-boundary/README.md)

[Research graph](../../../graph/index.html#node=2026-10-16-candidate-specific-prebind-refit-saves-603-cold-495-warm) · [Copy provenance](../../../PROVENANCE.json)

