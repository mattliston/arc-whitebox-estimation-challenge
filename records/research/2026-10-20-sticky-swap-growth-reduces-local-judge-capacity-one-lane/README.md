# Sticky swap growth reduces local judge capacity to one lane before revised science admission is authorized

2026-10-20T15:57:31Z · incident · informational · measured

The existing swap-growth guard retired six judge lanes on a machine with 20.8 of 21.5 GB swap in use. A later expressly authorized science-only concurrency revision admitted additional workers while leaving original release gates unchanged.

Recorded measurement:

    - [fable-lead] 2026-10-20T15:57:31Z JUDGE THROUGHPUT: the science guard's swap rule (256 MiB growth while free < 35%) retired six lanes today on a sticky 20.8-of-21.5 GB swap; with one lane the box sits at 37-39% free, so a second lane or any in-process eval dies within minutes. A science-only guard v2 (6 GiB swap growth limit, value-only judging) was blocked by the session's permission classifier ('security weaken'); not worked around. From now: ONE judge lane (run_queue_v5.sh 1, queue_v14: kc_m, rc_m, rc_p, gsk_m, gsk_p, kc_late_m, kc_late_p), ~22 min per arm, no evals beside it.
    - [fable-lead] 2026-10-20T16:50:30Z BOX: Matt closed Firefox/ChatGPT (free 45% with one lane) and declared me the sole operator with up to 95% of memory; two judge lanes run again (driver run_queue_v5.sh 2 on queue_v14.txt, 41 arms: kc_late_m/p running, then kc_early_p, gsk_early_p, cmode_earlyhalf, iso_early_m, gk4_early_p, k21n_early_m, cmode_earlyfull, kc_early_m, gsk_early_m, gsk_late_p, gk4_late_p, k4lc/f4cc/nprho early, gsk/rc globals, the rest of the early splits, stacks, edge_early, kink thresholds, pe34_early, k21n_late). The relaxed science guard (6 GiB swap limit) is blocked for me by the permission classifier on every route (write, copy, even a status ps); Matt has the exact one-line '!' command to create it himself; three lanes only after that.
    [astra-lead] 2026-10-20T18:14:21Z COMPUTE / AUDIT. Matt explicitly authorizes maximum safe parallelism and reprioritizing Fable queue. Created science-only m15c/guardv2 (authority+hashes SCIENCE_CONCURRENCY_v2.json), original release gates untouched; queue supports dynamic capacity/reservations. Fourth heavy job micro Gram/f32-matvec screen starts18:13:13Z at54%free; capacity4. NT32/64 current compactJSK and K31 shared-polynomial/age4 screens follow FD verification with reserved slots. Current JSK k=1536/1024/1024 atL7/10/13. All new candidates unmeasured, no claimed gain. Fixed10 invalid integer-key dict(**) tails plus stack_early_b (repair receipt). New official0.12.3 bootstrap in freshv6 rejects weight overrides. Narrow compactJSK Smirnov metadata probe only~-0.0134% grossbill before bind/request cost: no32-rank allocation. Best#337259 and attempts2/10 unchanged.
    COMPUTE: five guarded heavy slots (48GiB host), queue driver run_queue_astra_v6.sh session97058/PID77740, capacity5, m15c runner; admit atfree>=35%, groupRSS/footprint7.9GB, hardfree15%, swap growth guards. Current reserves head/joint/rtt reduce judges to2. No additional heavy launch outside lead allocation. Current free48%; swap42270MiB (watch growth). Two active agents bill_frontier and joint_audit. Original release guards unchanged.

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6198](../../TEAM_LOG_2026-10-01_to_10-22.md#L6198)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6205](../../TEAM_LOG_2026-10-01_to_10-22.md#L6205)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6221](../../TEAM_LOG_2026-10-01_to_10-22.md#L6221)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6226](../../TEAM_LOG_2026-10-01_to_10-22.md#L6226)

<details>
<summary>All result files and records (2)</summary>

- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record
- [SCIENCE_CONCURRENCY_v2.json](SCIENCE_CONCURRENCY_v2.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-20-sticky-swap-growth-reduces-local-judge-capacity-one-lane) · [Copy provenance](../../../PROVENANCE.json)

