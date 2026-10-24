# Depth re-pricing under memory pressure measures paging rather than kernel speed

2026-09-23T17:12Z · incident · informational · measured

In-call comparisons exposed millions of page faults and inflated backend time while user CPU stayed stable. Large training jobs were paused to obtain interpretable kernel measurements.

Recorded measurement:

    - 2026-09-23 10:12 [engineer] Depth re-pricing on backend time STARTED (no builds): in-situ paired tool fable_engineer_v1/depth_shadow_v3.py runs every push and read at every allowed depth inside the same call (random order, only the S7-rule output kept), so depth costs are paired per call under one load; 2 concurrent runs, warm nets 35,100,37,39,42,48 (a cold warm-up net first). Cold smoke on r35: 1024x384x8192 pushes (5/net) cost D4 1.13 s, D5 0.76 s, D6 2.62 s backend per call for bills 3.93/3.58/3.40 Gflop, so six levels look badly priced on thin k. Pareto + validation pairs to follow.
    - 2026-09-23 10:46 [engineer] DEPTH RE-PRICING DONE (no builds; fable_engineer_v1/RESULT.md E-46). In-situ paired tool (every push/read run at every depth inside the same call, S7 output kept; 6 warm nets 35,100,37,39,42,48). (1) Backend time on this box is currently PAGING: 1.9-2.9M faults and 21-39 s system CPU per warm net; backend 113-178 s/net while user CPU stays 54.6-56.1 s; six-level pushes take the most faults (their idle arena region gets compressed). Wiring the arena (mlock control) makes six-level backend = user CPU (18.7 vs 18.6 s/net, faults 16k vs 893k). Concurrent pairs double the footprint and can cause this themselves: check faults/system CPU before trusting a backend pair. (2) Compute pricing of push5/push5p6 was right: in-situ user CPU matches the old isolated model within ~20%/class; padded six (S7 - S6b rule) = +2.79 s user/net (+5.3%) for -0.457% bill (platform showed +5.0%). (3) Pareto vs S6b rule: best at compute <= S6b saves only 0.08-0.15% bill (S6b is on the frontier); MU 0.6: +0.22% bill, -2.75 s user; all <= five levels: +1.15%, -5.9 s. Depth trades at 0.5-0.9 Gflop per user-second; slices of push5p6: k>=992 -0.143% for +0.60 s, k>=928 -0.294% for +1.44 s. Symout v3 is several times better per wall-second, so no new depth rule now.
    - 2026-09-23 10:47 lead: MEMORY PRESSURE (swap 5.6 of 7.2 GB, compressor 8.5 GB, heavy paging): PAUSED both TRAIN640 refits with SIGSTOP (PIDs 69931 knobs, 47676 trunk). State kept; they RESUME with 'kill -CONT 69931 47676' after the S10 gates and VALID40 finish (~11:15). If a session restart finds them in state T, run that command.
    - 2026-09-24 10:01 lead: RESUMED both TRAIN640 refits (kill -CONT 69931 47676) after the usage-limit gap; memory 79% free.

## Programs

- [depth_shadow_v3.py](depth_shadow_v3.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L306](../../TEAM_LOG_2026-09-22_to_09-26.md#L306)
- [TEAM_LOG_2026-09-22_to_09-26.md#L307](../../TEAM_LOG_2026-09-22_to_09-26.md#L307)
- [TEAM_LOG_2026-09-22_to_09-26.md#L308](../../TEAM_LOG_2026-09-22_to_09-26.md#L308)
- [TEAM_LOG_2026-09-22_to_09-26.md#L310](../../TEAM_LOG_2026-09-22_to_09-26.md#L310)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log
- [RESULT.md](../../../experiments/arithmetic-kernels/grouped-winograd-requests-copied-operands-fewer-requests-but-no/RESULT.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-23-depth-re-pricing-under-memory-pressure-measures-paging-rather) · [Copy provenance](../../../PROVENANCE.json)

