# Depth-six contraction splitting saves 0.96% work; backend cost depends strongly on BLAS

2026-09-25T09:22Z · experiment · informational · measured

Splitting long contractions into fitting chunks enables deeper multiplication after padding. Measured OpenBLAS cost was much lower than the earlier Accelerate estimate, leaving a viable wall tradeoff.

Recorded measurement:

    - 2026-09-25 02:22 [engineer] E-72c (lead's k-split question, paper, RESULT.md top): depth-5 bank read/push products with inner dim >= 4096 = 25/net = 24.9% of S23's net bill (6-8k only: 19/net, 21.9%). Split k into chunks <= 4096 that fit S24's depth-6 regions, m/n padded to 64: bill -1.02% of the net bill (-0.98% with pad copies; 6-8k only -0.89%), +3.2k requests/net; padding eats most of the gain on 16 products (928x8128x992: -2.8% vs -9.5% unpadded). CPU EXTRAPOLATED from E-72 rates (depth-5 big-k 0.011 s/GF vs depth-6 0.035-0.06): class backend 3-5x = +1.7..+3.6 s/net (+11..+23%) = ~+8..+16 s on S24's worst MLP. Closes on CPU (2-4x worse per CPU % than m64); reopen only as a deliberate wall spend, first a chunked depth-6 bench. Tool: fable_engineer_v1/e72c_ksplit_price.py.
    - 2026-09-25 02:23 lead: DEPTH-5 K-SPLIT (engineer E-72c): the depth-5 bank read/push class with inner dims >= 4096 is 24.9% of the bill (25 products/net). Chunking to <= 4096 at depth 6 (m/n padded to 64, chunk outputs added) = bill -1.0%, but backend 3-5x on the class (+1.7..3.6 s/net local, ~+8..16 s platform worst) and +3.2k requests (~+21 ms). The wall is affordable (S24 worst 72.3 s vs the 100 s rule). GO: a chunked depth-6 bench on the 4 biggest products to pin the CPU band. Then price S26 as one package = k-split + shelved request offsets (group 5 -974, top-level fusion (a) -0.75k, E-73 -0.45k, j3 -0.17k): net ~+0.8k requests (~+5 ms) for ~-1.2% of bill.
    - 2026-09-25 02:43 [engineer] E-72d K-SPLIT BENCHED (lead GO; RESULT.md top; e72d_ksplit_bench.py / _v2.py, all 24 classes = 25 products/net, S24+E-73 in process): bill -2.97 GF = -0.96% of the net bill MEASURED (incl. padding). CPU under the OpenBLAS env (the wall law's instrument): class backend x1.63 (chunk keys unfused) .. x1.79 (fused) = +2.4..+3.0 s/net (+9.5..+11.8%); E-72c's 3-5x was an Accelerate (.venv) artifact (Accelerate: x3.8). Wall (project_wall_v3 on #332255, ew 70.7 s, Kw 63.75 s): fused +5.3/+6.9/+8.4 s, unfused +4.7/+6.2/+7.6 s. Requests +2,635 fused / +3,719 unfused per net. Setup: +21 prebind keys net (50 chunk keys in, 29 class-only depth-5 keys out) -> slowest platform workers reach the 3.0 s guard. CORRECTION to the package math: top-level fusion (a) is ALREADY in S24 (lvfuse2 fuses level 1 of both forward transforms; 359/362 keys fused); only a level-1 unwind pair remains (~-0.18k, not built). Package vs S25 = k-split fused + group 5 + unwind pair: bill -0.98%, +1.40k requests (~+9 ms), wall +6.4 s; vs S24 -1.16%, +0.78k (~+5 ms), expected worst ~77 s. Depth 7 re-checked under OpenBLAS: +38..+151% backend, stays closed.
    - 2026-09-25 02:44 lead: K-SPLIT BENCH (engineer E-72d, S24+E-73, OpenBLAS env): bill MEASURED -0.96% (padding incl.). Class backend 1.63x unfused / 1.79x fused = +2.4..3.0 s/net (the earlier 3-5x was an Accelerate artifact). Wall +5.3..8.4 s on S24's expected worst 70.7 s. Fusion (a) is already in S24 (lvfuse2); only a level-1 unwind pair (-0.18k) is left. Package vs S24 (with E-73 + j3 + group 5): bill ~-1.16%, +0.78k requests (~+5 ms), expected worst ~77 s.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_engineer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/e72c_ksplit_price.py](code/e72c_ksplit_price.py)
- [code/e72d_ksplit_bench.py](code/e72d_ksplit_bench.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L945](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L945)
- [TEAM_LOG_2026-09-22_to_09-26.md#L946](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L946)
- [TEAM_LOG_2026-09-22_to_09-26.md#L968](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L968)
- [TEAM_LOG_2026-09-22_to_09-26.md#L969](../../../records/TEAM_LOG_2026-09-22_to_09-26.md#L969)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../../records/TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-depth-six-contraction-splitting-saves-0-96-work-backend) · [Copy provenance](../../../PROVENANCE.json)

