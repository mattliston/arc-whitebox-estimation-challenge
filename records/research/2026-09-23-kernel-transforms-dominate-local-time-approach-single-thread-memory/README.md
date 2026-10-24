# Kernel transforms dominate local time and approach single-thread memory bandwidth

2026-09-23T07:14Z · finding · informational · measured

Call-site profiling showed that additions, subtractions and copies dominated kernel time. Copy elimination had to be judged against the remote-request cost it introduced.

Recorded measurement:

    - 2026-09-23 00:14 [engineer] TIME PROFILE S3-HE r35 cold (fable_engineer_v1/time_profile.py: in-process with the server's numpy 2.4.6/OpenBLAS, 1 thread, load 12-15): numpy 50-55 s = kernel 40-43 s (Winograd add/sub 26-28 s, leaf GEMMs 10.4-10.7 s, copies 6-7 s) + body 10-13 s (top: _nonlinear_pair_batch multiply 2.0 s, _predict_binary multiply 1.3 s). Transforms move ~0.5 TB/net at 20-35 GB/s = near single-thread memory bandwidth: bit-exact kernel cuts must remove traffic; level-(L-1) copy elimination saves ~2.4 s local but costs ~+30k requests (~+6.6 s platform) -> no. HEB interleaved x2 vs HE: copies -1.4 s, leaf GEMMs +1.1 s -> compute-neutral, still -0.35% bill bit-exact. -5 s bit-exact is out of reach in the kernel.
    - 2026-09-23 00:30 [engineer] CALL-SITE TIME PROFILE S3-HEB r35 (fable_engineer_v1/product_profile.py -> product_profile_S3HEB.json; in-process, server numpy, 1 thread, load 14-15 so ~1.5x quiet seconds): numpy 73.8 s = kernel 53.3 (72%) + body 20.5; r100 warm 38.6 + 13.1. By product class r35: six-level pushes 22.2 s = 30% of numpy for 26.6% of bill (4.8 Gflop per compute-second), five-level reads 14.7 s / 29.4% bill, 1024^3 D5 squares 10.2 s / 23.8%, four-level pushes 2.8 s / 8.3% (12 Gflop/s). Top body sites (engineer2): _predict_binary:3240-3244 float64 diagonal 3.9 s (5.3%), _nonlinear_pair_batch:906-919 3.75 s. Top-20 table sent to lead + engineer2. Next: compute-budgeted depth reallocation (new five-level push layout, fable_engineer_v1/make_push_layout.py, reproduces the W-F D6 layout exactly; bench depth_bench.py running).

## Programs

- [time_profile.py](time_profile.py)
- [product_profile.py](product_profile.py)
- [make_push_layout.py](make_push_layout.py)
- [depth_bench.py](depth_bench.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L111](../../TEAM_LOG_2026-09-22_to_09-26.md#L111)
- [TEAM_LOG_2026-09-22_to_09-26.md#L122](../../TEAM_LOG_2026-09-22_to_09-26.md#L122)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [product_profile_S3HEB.json](product_profile_S3HEB.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-kernel-transforms-dominate-local-time-approach-single-thread-memory) · [Copy provenance](../../../PROVENANCE.json)

