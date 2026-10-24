# Remainder-32 strided-read variant is byte-identical but does no work on the measured network

2026-10-13T01:59Z · experiment · inconclusive · measured

The first native metered FLOPs cut from the independent hunt, qz7_native_v2 (a variant of the grouped strided-read kernel seven-level strip push (HZ7) for products whose dimension is 32 mod 128), reproduced the submitted stack bit for bit on network 7100136 but touched nothing there: no product with m or k = 32 mod 128 occurs on that network, so metered FLOPs and requests were unchanged and nothing could be integrated until a network with remainder-32 shapes shows a lower metered FLOPs.

Recorded measurement:

    pred sha 5eb24347 identical, 16/16 layer MSEs equal; bill +0, requests +0; _QZ7_STATS products 0 / packets 0

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/opus_reexam_20261001_v1/scout_native/build_byteid_stack_v1.sh](code/opus_reexam_20261001_v1/scout_native/build_byteid_stack_v1.sh)
- [code/astra_hunt_20261013_v1/qz7_v1/candidate.py](code/astra_hunt_20261013_v1/qz7_v1/candidate.py)
- [code/astra_hunt_20261013_v1/winograd_review/check_qz7_v1.py](code/astra_hunt_20261013_v1/winograd_review/check_qz7_v1.py)
- [code/astra_hunt_20261013_v1/winograd_review/fragment_qz7_v1.py](code/astra_hunt_20261013_v1/winograd_review/fragment_qz7_v1.py)
- [code/opus_reexam_20261001_v1/analyst/fresh/loader.py](code/opus_reexam_20261001_v1/analyst/fresh/loader.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/fastgate_post_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/fastgate_post_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py](code/opus_reexam_20261001_v1/engineer/a62_modules/make_s49_v1.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k1s_pbs/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k1s_pbs/candidate.py)
- [code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k2tw_x/candidate.py](code/opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k2tw_x/candidate.py)
- [code/opus_reexam_20261001_v1/release/guard_footprint_v3.py](code/opus_reexam_20261001_v1/release/guard_footprint_v3.py)
- [code/opus_reexam_20261001_v1/release/rusage.py](code/opus_reexam_20261001_v1/release/rusage.py)
- [code/opus_reexam_20261001_v1/scout_native/run_cert_v2.py](code/opus_reexam_20261001_v1/scout_native/run_cert_v2.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1874](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1874)

<details>
<summary>All result files and records (4)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [qz7_native_v2_cert.json](qz7_native_v2_cert.json) · result
- [qz7_native_v2_cert_comparison.json](qz7_native_v2_cert_comparison.json) · result
- [qz7_native_v2.log](qz7_native_v2.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-10-13-remainder-32-strided-read-variant-byte-identical-but-does) · [Copy provenance](../../../PROVENANCE.json)

