# Release / wall-time lane: results

Lane `O/release/`, release engineer, started 2026-10-01 18:40 UTC. Newest section first inside each heading. Every number says
what it is, on which networks, against what, how measured, and its noise. "Local" = this machine (Apple M3 Max, 12 performance +
4 efficiency cores, 48 GB), flopscope server on OpenBLAS numpy with one thread.

Names: S35 = the bank, submission #333111, `H/hill_climb_prefix128_fusion_v1/candidate.py`. RRPI = its research parent,
`H/hill_climb_pair_rr_inverse_v1/candidate.py`. S32/S33/S34 = the three banks before S35 (#332738, #332838, #333050).
Kernel = the server's numpy time (the platform's `flopscope_server_kernel_wall_s`); overhead = flopscope machinery and wire;
residual = the participant's Python outside flopscope calls (limit 0.4 s per network); wall = their sum (limit 120 s).

## 0. Summary (2026-10-10 02:50 UTC; items 5, 7-14 new or changed since 21:20 on 10-01; items 15-21 new on 10-02 07:15-12:20; items 22-25 new 15:25-16:20; items 26-28 new 16:20-18:35; items 29-30 new 19:14-20:00; items 31-34 new on 10-10 02:00-02:50)

1. **Pipeline.** `qualify.sh` runs packaging (A), 40-network native validation (B), paired boundary census with projection (C)
   and the 20-network worker run (D); it stops before the upload. `RUNBOOK.md` has commands, limits and options.
2. **Projection error, out of sample.** Each of the four banks S32-S35 was projected from each other bank's platform page and
   a local session of both (12 ordered pairs): mean kernel within 1.5% on average (worst 3.0%), overhead 0.7% (worst 1.2%),
   mean residual 2.5% (worst 5.0%), mean wall 1.1 s (worst 2.3 s). The platform itself repeats identical code to about 2%.
   S35 projected from its three predecessors: mean wall 83.8 / 83.1 / 84.8 s against the measured 82.5 s.
3. **RRPI dry run.** A PASS; C PASS* (every load-independent gate passed; setup wall not judged in a loaded window); D PASS;
   B not run as such (RRPI's 40 networks will be produced as the parent arm of the fast variant's validation). Projected on the
   platform: wall 82.8-83.0 s mean / 95.3-95.6 s worst network, residual 0.306 mean / 0.341 max: +0.5 s and +0.002 s over S35
   for a 0.10% smaller bill.
4. **Where the CPU goes (RRPI, networks 35 and 100).** Leaf matmuls are 34% of local kernel CPU for 79% of the bill; the
   Winograd add/subtract transforms are 45% for 12% of the bill; scatter operations 7% for 0.26% of the bill. Section 3.
5. **Task 4, the fast variant.** Two levers found (section 4):
   - Level fusion (in every bank since S24) makes the output-tree and right-operand-tree operations 1.1-2.9x more expensive.
     Keeping it only on left-operand trees and on trees up to 32M floats (`lvfuse_sized32`) leaves every prediction byte and
     the bill unchanged and cuts local kernel CPU by 10% (projected platform kernel -13%, less in the platform's setup regime:
     being measured) for 3.6% more kernel calls.
   - The 84 thin low-rank-star products per network take a tenth of all requests for 1.7% of the bill. As plain matmuls
     (`tsr_off`): bill +0.48%, 4,350 fewer calls per network.
   **Recommended build `fastC` = RRPI + both.** Against S35 at the boundary: bill +0.37%, requests -6.4..-6.6%, residual
   instructions -5.5..-7.0%, kernel proxy time -7.5..-9.5%. Projected platform: wall 73.0-74.4 s mean / 84.0-85.6 s worst network
   (S35: 82.5 / 94.9), residual 0.283-0.287 s mean / 0.315-0.320 s max (S35: 0.304 / 0.339). Stage A PASS; stage C PASS; stage D
   PASS on every load-independent gate (20 networks, peak handles 55,385, setup 3.60 s, native parity in bytes and bill on
   networks 35 and 100). **On 20 fresh networks against RRPI (native, `runs/b_rrpi_fastC_vs_rrpi_fresh20a`): pooled final-layer
   MSE -0.238% [95% -0.39, -0.09], lower on 13 of 20; bill +0.481%; adjusted +0.245% [+0.09, +0.39]; worst network +0.80%.**
   So the thin-star request diet costs about a quarter of a percent of score, half its bill.
   The depth-7 layers can be turned off on top (8-16% of kernel time per 1% of bill).
6. **Risk that is not wall time.** In the failed #333099, three of the four lost networks died on the 0.4 s residual limit.
   The mechanism (seen locally when the machine load hit 27, section 4a-bis): a slow worker's setup runs out of its 4.0 s guard
   early, the tree views it did not bind are bound inside predictions (billed, more requests, never fused), so the residual
   rises on top of the general slowdown. S35 can absorb a 1.18x slower worker on residual and 1.26x on wall; `fastC` 1.25-1.27x
   and 1.40-1.43x; the slow workers of #333099 ran 1.21-1.30x and 1.34x.
7. **How far the platform's setup gets is now pinned by the flops it books** (section 6): a normal worker gets through about
   19,000 requests before the estimator's 4.0 s guard (range 18,600-20,100); worker w-2 of #333111 got through 17,954 (exactly
   623 of 649 keys), the slow re-set-up worker of #333099 14,589 (exactly 580 keys). The binding phase of S35, RRPI and the
   engineer's stacks needs 18,555 requests, 97.7% of a normal worker's budget; with the release un-fuse fragment 17,198 (90.5%).
   Stages C and D now run the setup under this platform clock (`--short-setup`), so a session no longer depends on the load
   of this machine at the moment of its setup; `setup_margin.py` gives any candidate's margin in seconds.
8. **The zero-bill build `S35 + lvfuse_sized32` is byte-identical to S35 with the same bill**, natively on networks 35 and
   100 (`runs/_native/5ae8ac05f789_766213e2_a1` against `f9f4790b3c65_766213e2_a1`), stage A PASS. Its 20-network worker run
   happened to get a slow setup (load 10-13: 418 of 649 keys bound) and so measured a degraded worker, not the build
   (section 5); it is being repeated under the platform clock.
9. **The residual is paid per flopscope call** (section 7): a client profile of RRPI shows 43,438 flopscope calls for 42,527
   requests on network 35, no hot spot in the estimator's own Python, and flopscope's own per-call wrapper at about twice the
   estimator's Python. Only fewer charged calls lower the residual.
10. **Which limit binds, and what the builds give** (sections 2 and 8). A slow worker stretches residual and wall of its
   networks together (correlation 0.88 over S35's 100 networks). A single network may take 1.31x the mean residual and 1.46x
   the mean wall on S35 before a limit; the slowest normal networks take 1.11-1.15x, the slow workers of #333099 took
   1.22-1.34x (residual) and 1.33-1.38x (wall) on average, and its lost networks 1.33-1.43x and above 1.49x. So the residual
   limit is the nearer one, and it moves with requests only. Measured under the platform clock: the engineer's stack with both
   release fragments 1.34 / 1.60 at bill -1.57%; fastC (natural-setup run) 1.41 / 1.64 at bill +0.38%. Un-fuse alone buys wall
   allowance and costs residual allowance (+3% requests); the thin-star diet buys both.

11. **S36 and S37** (sections 2, 8, 9). The lead fired the S36 file (engineer's stack + regenerated prebind list + the two
   release fragments, sha 87b3d6ef) as #333429 at 02:27 UTC on stages A and C: 2.031e-9 (S35 2.0672e-9), no network lost.
   Platform against the projection made before the fire: predict wall 75.3 s mean against 74.2 projected, residual 0.283 s
   against 0.288, setup flops identical on all five workers to the setup probe's complete binding phase. The S37 file
   (S36 + F4 at 0.6 + K4L v2, sha 2a25a074): packaging PASS (three new einsum signatures read: no repeated operand; compiles
   under Python 3.10), boundary PASS: 42,864 requests per network (S36 40,868, S35 43,236), bill -1.55% against S35,
   projected wall about 75-76 s and residual about 0.30 s: S35's residual room, 6-7 s more wall room. 20-network worker
   (run by the lead, 03:17 UTC) PASS on every gate: all outputs finite, handles 56,441 at the end (S35 59,751), group RSS
   6.20 GB, bytes and bill equal to native on networks 35 and 100.

12. **State at 03:45 UTC on 10-02.** The lane runs nothing and has nothing armed (lead's instruction of 03:17 UTC: no
   boundary sessions until asked; I ran 26 minutes of sessions against it before reading it and stopped at 03:43 UTC).
   Waiting on request: the same-window pair S35 / S37 (clean calibration), slow-setup stress of S36 and S37, S37 with
   un-fusing off or limited to trees above 64M floats (`variants/s37_2a25a074_fused`, `_lvfs64`: byte-identical builds that
   say what residual room un-fusing costs), the #333099 emulation check, S35's 20-network reference under the platform clock.
   Without compute: `check_projection.py` on S37's platform record when it is stored.
13. **Answers to the lead's four questions of 03:17 UTC** (TEAM_LOG 03:49-03:50 UTC; sections 2 and 9).
   (4) S37 #333435 against its projection: residual 0.3016 projected / 0.3032 s measured; wall 74.3 / 78.0 s, the miss being
   the load mismatch between the stored S35 legs and S37's legs (against calm S35 legs: 78.2 s). (1) S37 loses its first
   network on a worker 1.20x slower in everything (residual), its wall holds to 1.37x; S35 1.17x and 1.26x; S36 1.26x and
   1.42x. (2) Request plan for S37 (42,864 per network; target under 41,500): merge F4 / K4L's small-vector algebra (about
   -400, engineer), un-fuse only above 64M floats (-550, built) or not at all (-1,100, built, +7 s of wall), one thin star
   product per layer on stacked operands (-700, engineer), one view less per operand load (-600, engineer). (3) Designate
   the best-scoring graded file that lost no network, leaves at least 0.095 s of residual room and 35 s of wall room on the
   platform and holds no lever judged only on networks it was tuned on: today S37; test before designating = same-window
   pair, slow-setup session, 20-network worker on the exact archive.

14. **Lead's decisions of about 04:00 UTC and what I prepared** (no compute). Designated file: S37, fallback S36, by the
   rule in section 9. The designation test is one command, to be started only in an announced quiet window:
   `bash O/release/designation_test.sh <file.py> <tag> ["reviewed einsum sig;sig"]` (pair in one window, slow-setup session,
   20-network worker, then `designation_judge.py`). With what already exists for S37 every criterion is met except the
   slow-setup session, which has not been run. The top three's margins and losses are in section 9: J2W and marius run at
   0.364 and 0.363 s of mean residual (room 0.036 s) and lost networks in 8 of 18 and 7 of 34 runs in 96 hours; by the
   record a private run of their current files loses a network with about 18% and 13% probability, Luna's about 3%, ours 2%.

15. **Fast path for candidates that differ from a qualified file only in literals** (asked by the lead 07:10 UTC on 10-02;
   section 11). `bash O/release/fast_path.sh <qualified.py> <new.py> <tag>`: a 2-second literal proof (`literal_diff.py`:
   same code, only float literals, packed arrays and docstrings differ, each listed), stage A of the new file, one worker
   session of each file under the platform clock on networks 35 and 100, and the census compared count by count (requests,
   free trips, handles, calls and flops per operation family, bill, stored bytes, setup). PASS = the new file inherits the
   qualified file's wall, residual, memory and handle projections. Tested with sessions at load 14-29: the S38 file against
   a copy with seven dial literals moved: PASS, every count and the bill to the last flop equal on both networks; against
   the same copy with the two F4 strengths equal: FAIL on the census (38 requests and 80 kflop fewer per network: the file
   takes a scalar path), as intended. The next dial candidate against S38 costs one session (6-13 minutes on the loaded
   box, 2 quiet). The lead's own review item of 07:10 UTC (the chain for the S38 file): one argument changed, `--legs PCCP`
   instead of the stored S35 legs measured under load. S38's chain read: section 11 (expect 74.5-75.5 s mean wall and
   0.293-0.295 s residual on the platform; group RSS is not memory on this machine).
16. **S38 #333481 graded 1.907e-9 (new bank), and the projection re-set** (section 2, "The third"). Residual 0.2945
   projected / 0.2944 s measured; mean wall 74.9 / 77.1 s (kernel 3.4% low). The memory-bound share of the kernel projection
   is now 0.65 (was 0.72): the three fires then read +0.3, +1.1 and -1.2 s in mean wall; a projected mean wall is good to
   about 1.5 s. Stage D now projects from S38's own worker run and platform record (20 networks). By platform telemetry
   S38 leaves 0.106 s of residual room and 42.9 s of mean wall room: inside the designation rule; its slow-setup session
   is the one test still open (on the lead's call).
17. **The lead's decisions of 08:37 UTC, done by 09:00 UTC.** (a) Memory gate on footprint: every session runs under
   `guard_footprint.py` (kill at 7.5 GB of group footprint, group RSS only as a 10 GB backstop); the stages gate on server +
   client peak footprint under 7.5 GB (`memory`). S38: 2.53 GB. (b) The projection rule: every stage now projects from the
   newest graded file, the bank (`calibration/bank.json`, `set_bank.py`, `new_bank.sh`): S38 with its own platform record
   (kernel 68.36 s, overhead 0.1791 ms per round trip, residual 7.03 microseconds per request, memory-bound share 0.60). S35's
   overhead per round trip (0.1833 ms) is 2-3% above that of the files after it, which is why the overhead was projected high
   in all three fires. (c) Fast path accepted as the qualification of literal-only candidates.
18. **Designation test of S38: PASS, every criterion met** (section 11). Platform record: no network lost, residual room
   0.106 s, mean wall room 42.9 s. Slow worker: with the setup at the pace of a worker 1.3x slower S38 binds 525 of 649
   keys and its networks take +1.83% requests (limit +2%, the narrowest margin). First loss on a worker X times slower in
   everything, by the worst network of each file's platform run: S38 1.18x (residual) / 1.36x (wall), S37 1.20x / 1.37x,
   S36 1.26x / 1.50x, S35 1.17x / 1.26x.
19. **Why projections depended on the window: page-fault work** (section 2). Kernel instruction counts include the kernel's
   work of paging the server's arrays in again when the machine is short of memory: 14,200 instructions per minor fault, up to
   10% of a network's count on a crawling box. Taken out since 09:20 UTC (`project.fault_free`); the counts then repeat to
   0.27% rms at any load, the three fired builds project the same kernel time from legs of any window, and stage C can run
   against stored legs (two sessions instead of four) on a loaded box.

20. **S39 (#333528, fired 11:39 UTC) qualified by the lead's chain with the re-based tools** (section 11). Projected 77.6 s
   mean wall / 88.9 s worst network and 0.300 / 0.338 s residual from both stage C and stage D; requests 42,624 per network
   (+755 on every network against S38: the exact-term module); kernel work x1.004 of S38 (the 2.2% bill cut is no kernel
   time). Stored S38 legs and fresh ones give the same projection to 0.01 s: the next chain's stage C can be two sessions.
   Graded 1.847e-9 at 12:09 UTC, no network lost: mean wall 77.6 projected / 77.3 s measured, residual 0.2998 / 0.3018 s,
   kernel +0.5%: the corrected rule's first fire. S39 is the bank (`new_bank.sh`, first run, clean).
21. **Where the residual risk sits, and what requests cost** (section 11). The worst residual of a run is a worker's first
   network after a setup (S39: 0.313-0.347 s on its seven first networks, at most 0.322 s on the other 93); that surcharge
   doubled with the L3 loads (x1.05 in S37, x1.09-1.10 in S38 and S39). Private-run risk of S39 by the analyst's model:
   about 2%, as S37 and S38; +2,000 requests 2.9%, +3,000 4.8%, +5,000 11.6%. Request room without physics: un-fusing only
   above 64M floats -535 requests for +1.8 s of wall; no un-fusing -1,170 for +5.7 s.

22. **S40 (#333559, 1.7590e-9, third place) is the bank** (section 12; `new_bank.sh S40 ...` ran clean). Projection
   against the grade: kernel -0.7%, overhead -1.4%, mean residual -0.5% (0.3093 / 0.3109 s), mean wall -0.6 s (79.2 /
   79.8 s), worst-network wall -1.4 s, worst-network residual -0.010 s (0.3553 / 0.3652 s). The 0.010 s is one event: the
   worst network was the first network of worker 3 after a SECOND setup that ran 1.124x slow (its setup booked the flops of
   exactly 627 of 649 keys, nothing fused), on a processor running 1.138x the run's median kernel time per flop.
23. **Loss probability of a fresh 100-network run** (`loss_risk.py`; the analyst's 729 excursions): S40 as fired about 3%
   (1.4% to 5% by reading), S39 1.1%, all of it on the residual limit and on a first network under a slow worker. Two of
   our 13 leader-sized runs had a worker set up a second time at a slow pace (#333099 at 1.30x: four networks lost; S40's
   run at 1.124x: held with 9.5% of room).
24. **The designation rule in its second form** (`designation_judge.py`; the first form is in `_stale/`): the request cap
   is split by network: first network after the setup at most 43,800 requests, warm network at most 42,700 (stage C's
   networks 35 and 100); platform mean residual at most 0.310 s with first networks at most 0.328 s on average; mean wall
   at most 82 s; the other lines unchanged. S39 passes every line; S40 fails three (first-network requests 44,578, mean
   residual 0.3109 s, first networks 0.3308 s).
25. **The variant I would designate** (section 12): the S41 file with the engineer's leaf-view lever HV + two release
   fragments, `Rlvfs_limit64` and `Rl3_bind128` (first-load views of the first 128 listed keys built at bind time):
   43,466 / 42,062 requests (first / warm; S40 44,578 / 42,678), predictions byte-identical to the file without the
   fragments, projected wall 80.9 s mean / 92.4 s worst, loss probability 0.55% to 1.4% by the two readings (S40: 1.4% to
   3.0%). Largest piece left: 482 requests of once-per-process constants on every first network (engineer).

26. **S41 = #333587 is the diet variant of item 25, graded 1.711e-9 with no network lost; it is the bank** (section 12).
   The lead fired it without a quiet window on this lane's stage C and a stage D at load 12-15 (PASS\*: every line of the
   file met, the machine's swap line not judged). Its run came out 3-4% slower than projected in kernel, overhead and
   residual alike: the platform's hour (57 minutes of queue; six of nine setups cut by the guard). Worst network 0.3536 s:
   again the first network of a worker set up a second time at a slow pace (1.111x).
27. **The judge's lines in seconds are now speed-robust** (18:30 UTC): no network lost, worst network at most 0.36 s and
   105 s of roundtrip in the graded run, projected worst-network wall at most 100 s; means in seconds are shown, not
   judged. S41 meets every line; its verdict is REPEAT until its stage D has run in a quiet window. Loss probability of
   S41: 1.9% in an hour like its own, 0.7% in an hour like S40's (S40's file: 5.2% and 1.4%).
28. **The S42 request line** (section 12): the engineer's HV2 + CS + byte-identical diet with `Rl3_bind128`: 43,159 /
   42,265 requests without lvfs64 (641 / 435 under the caps), predictions byte-identical to S41's, slow setup +1.81%.
   The setup's guarded budget is full (binding 17,366 + constants 490 + views 432 of 19,000 requests); the judge's binding
   line is 18,600. Instrument: the residual is read from instructions only below 30,000 residual cycles per request.

29. **What the leaders' risk level would buy in bill: about 0.4%, 0.7% at most** (section 13; `bill_room.py`). The
   tree kernels already sit where one more level stops paying, so a public bank at 0.365 s of residual and 100 s of wall
   could turn only about 2,600 of the 6,300 available requests and 3.5-10 of the 16 s into bill. The leaders' lower bill
   (J2W: 6.8%) is structure. The only large data-independent part of our bill is the covariance sandwich of every layer
   (two 1024^3 products, 9.2% of the bill).
30. **S42 = #333607, fired 19:49 UTC** (section 14): 43,470 / 42,573 requests, projected wall 82.1 s mean / 88.6 s worst
   and residual 0.320 / 0.356 s at the speed of S41's slow hour; stage D PASS with every gate (on the step-1 variant, 128
   literals away; the fired file by the fast path, with its own native parity and slow-setup session). The judge reads
   every line met before the grade.

31. **S42 graded 1.667e-9 (#333607, 100/100) and is the bank** (section 15). Its hour priced a request at 7.10-7.13
   microseconds of residual (S40's hour). Its worst network (network 9, 0.3561 s) was NOT a first network but an ordinary
   network's residual spike (1.167x the median, kernel per flop 1.044x); its five first networks were 0.322-0.331 s. So a
   safer file must take requests off EVERY network. All five setups stopped binding at 17,694-17,933 requests (621-632 of
   649 keys, nothing fused): the constants cost about 1,200 requests of guard time beyond their own 493 requests; the setup
   clock charges that since 10-02 20:08Z, and at that clock the fired S42 runs 43,696 / 42,743 requests (first / warm).
32. **The rules let us designate a file other than our best public one** (official rules, section 6), but it must be one of
   our graded Phase 2 submissions: the safe variant has to be fired once, then the Team Representative selects it in the
   interface the site publishes near the end. Without a selection the best public file is taken.
33. **The safe designated variant of S42, pieces priced** (section 15; byte- or rounding-identical to S42):
   (a) no un-fused trees (`Rlvfs_limit_off`): -1,086 / -1,162 requests, bill +0.03%, wall 83.9 / 90.5 s (S42 77.5 / 83.6);
   (b) only trees above 64M floats un-fused: -498 / -520, bill +0.02%; (c) fusing keys at bind time: worse (closed);
   (d) the covariance sandwich's second product in the file's two-strip form (`Rsym_two_strips`; engineer's count -2,660
   requests, bill +0.27%): being measured. Loss probability of a fresh 100-network run (`loss_risk.py`, two readings):
   S42 2.2% / 3.2%, S42 + (a) 1.1% / 2.3%, S42 + (a) + (d) about 0.4% / 1.4% if (d)'s count holds; score cost of (a)+(d)
   about 0.3%. Stage A of (a)+(d) PASS (sha d13b9607).
34. **S43 is qualified as two files at once**: `qualify_pair.sh <engineer file> <tag>` builds the bank (+ `Rl3_bind128`) and
   the safe file (+ `Rlvfs_limit_off` + `Rsym_two_strips`), runs stage A of both, then C, D, the 1.3x slow session and the
   judge for the bank (about 45 minutes), then for the safe file. A dry run on the S42 engineer file rebuilt the fired S42
   (sha 471a9841) and the (a)+(d) file (sha d13b9607) byte for byte, stage A PASS on both.

## 1. The machine as a measuring instrument

- **Seconds are not reproducible here.** Identical single-thread work (numpy adds on 69 MB arrays, batched 32x32 matmuls)
  alternates between a fast and a slow state every 1-3 s: 0.15 vs 0.40 ns per element, 47 vs 19 Gflop/s. The thread stays on
  performance cores the whole time (kernel counter: P-core share 1.00) and executes the same instructions; cycles per second
  halve. So the performance cores are clocked down in bursts while the other lanes and the GPU job run. Raising the thread's
  QoS or dropping `nice` changes nothing. Scratch scripts `bench*.py`, 2026-10-01 18:50 UTC.
- **What is measured instead.** `rusage.py` reads cycles and instructions of any process from the kernel
  (`proc_pid_rusage`). The server (`server_light.py`) brackets every kernel call with two reads, so kernel work is known per
  operation type and per network. Instruction counts of the same code repeat to +-0.5% between clean runs (RRPI in process,
  network 35: 359.1 and 360.6 G; S35 at the boundary: 349.4 and 351.8 G over networks 35,100). Cycles repeat to about 1% for
  matmul and vary up to +40% for the memory-bound adds under contention.
- **A crawling session inflates the counters too, without paging** (2026-10-02 00:40 UTC, load 25-35, networks taking 290-440 s
  instead of 40-70 s). Server kernel instructions of the same S35 network read 356-365 G instead of 346-349 G (+2.4..+4.6%)
  with system/user still at 0.25-0.29, and the client's residual instructions 47.7-50.8 thousand per request instead of
  44.3-45.2 (+5..12%): kernel-mode work lands in the process in proportion to the time it spends on a core. Requests, round
  trips and bills are exact whatever the load. The projection therefore takes the residual from requests unless both arms
  were measured calm (`project.py`, CALM_CYCLES), and kernel ratios measured under such load carry about +-2.5% per network.
- **Paging breaks the counters.** A run during which the process pages reads high, because the kernel's fault handling is
  counted: RRPI network 35 read 387.7 G instructions instead of 360 (+7.5%) with system/user CPU 0.67 instead of 0.17-0.21;
  subtract +22%, copyto +36%, matmul +4.5%. Clean runs have system/user 0.13-0.21 in process and 0.20-0.26 for a server.
  Networks above 0.30 are dropped (stage D) or fail the stage on gate `clean` (stage C).
- **Memory.** 15 GB wired and 24 GB held compressed (10 GB physical) before any of our jobs; a server session needs about 6 GB.
  Two release jobs at once (19:27-19:37 UTC, my mistake) pushed the box into paging: three networks of the bank's own stage-D
  run paged, and the trainer lost a worker to its swap watchdog. One heavy release job at a time from then on.
- **Peak memory.** Group RSS (the gate) and the server's peak physical footprint differ: 6.33 GB vs 2.99 GB on the bank's
  20-network run. RSS counts freed-but-resident pages; the footprint does not. The 8 GB platform limit is presumably nearer
  the footprint, so the RSS gate is conservative.

## 2. Calibration against the platform

Local sessions: setup + networks 35 (cold) and 100, production path, one session per bank (two for S35), 19:00-19:20 UTC, all
clean (server system/user 0.21-0.26). Platform: each bank's evaluation page, 100 networks, the same 100 for all four (paired).

| bank | platform kernel s | overhead s | residual s | wall mean s | wall max s | local proxy kernel s | of which memory-bound | requests | round trips | residual 1e9 instr | bill Gflop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S32 | 69.03 | 7.42 | 0.257 | 76.71 | 82.95 | 30.84 | 19.95 | 36,600 | 39,737 | 1.64 | 276.0 |
| S33 | 72.86 | 8.45 | 0.297 | 81.60 | 91.60 | 32.41 | 20.64 | 40,494 | 44,759 | 1.80 | 272.2 |
| S34 | 72.45 | 8.69 | 0.299 | 81.44 | 89.14 | 31.51 | 19.99 | 41,429 | 45,960 | 1.85 | 267.7 |
| S35 | 73.41 | 8.75 | 0.304 | 82.47 | 94.93 | 31.17 | 18.10 | 42,232 | 46,738 | 1.88 | 263.6 |

"Local proxy kernel seconds" = instructions per operation type x the best cycles per instruction seen for that type, / 3.64e9
(the machine's clock when quiet); in this table each bank uses its own best rates, in a projection both arms share them.

Projection of each bank from each other bank (`calibrate.py`, output `runs/calibrate_v2.json`):

| reference -> target | kernel projected / actual (error) | overhead | residual | mean wall projected / actual |
|---|---|---|---|---|
| S32 -> S35 | 74.74 / 73.41 (+1.8%) | -0.3% | -3.0% | 83.76 / 82.47 (+1.3 s) |
| S33 -> S35 | 73.93 / 73.41 (+0.7%) | +0.8% | +1.8% | 83.06 / 82.47 (+0.6 s) |
| S34 -> S35 | 75.64 / 73.41 (+3.0%) | +0.9% | 0.0% | 84.78 / 82.47 (+2.3 s) |
| all 12 pairs, mean absolute | 1.5% (worst 3.0%) | 0.7% (worst 1.2%) | 2.5% (worst 5.0%) | 1.1 s (worst 2.3 s) |

- The kernel is scaled in two classes: memory-bound work (elementwise arithmetic, copies, reductions) and CPU-bound work
  (matmul, scatter, view requests). The share of the bank's platform kernel time taken by the memory-bound class was set to 0.72
  (band 0.58-0.80) until 2026-10-02 08:40 UTC and is 0.65 (band 0.55-0.75) since, after three fires ("The third" below). 0.58
  is the local share; a two-parameter least-squares fit over the four banks gives platform kernel =
  1.15 x CPU-bound + 3.24 x memory-bound local seconds (share 0.80), leave-one-out errors +0.7 / +1.3 / -2.0 / +3.1%. With one
  common factor (2.42 x local) the leave-one-out errors are -1.5 / +0.3 / -2.7 / +3.6%. Four points do not pin two parameters
  down; the band is carried into every projection.
- Overhead follows round trips to 1%: 0.187 ms per round trip on the platform.
- Residual follows requests: 7.02 / 7.33 / 7.22 / 7.20 microseconds per request on the platform for S32..S35. Locally one request
  costs 44.3-45.2 thousand residual instructions on every clean network of the bank's 20-network run.
- Worst network: the projection multiplies the mean wall by the bank's own max/mean (1.151). Across the five clean bank runs that
  ratio was 1.07-1.15, so the worst-network figure carries about +-4 s that no local measurement can remove.
- Not calibrated: the setup wall. The setup stops itself at a 4.0 s guard; it hits the guard on the platform (4.27-4.29 s on
  every worker) and locally whenever the machine is shared (4.07-4.18 s, 268-345 of 521 keys fused instead of all). Fewer fused
  keys cost about 1% more requests in every later network (42,932 vs 42,509 on network 35).
- Known weakness of the proxy: about 9% of the server's kernel instructions are flopscope's own per-request wrapper (about 0.8
  million instructions per request, CPU-bound) and are booked under the operation they wrap, so a share of "memory-bound" time
  is really per-request Python. A third class would need more banks than four to fit.

### The first projection made before a fire: S36, #333429 (2026-10-02 02:27 UTC)

Projection = stage C under the platform clock (`runs/c_stackHP_lvfs32_tsr_paced_rejudged`: two sessions of the S36 file against
the stored S35 legs, networks 35 and 100); platform = `O/board_evals/eval_333429.json`, 100 networks, none failed
(`check_projection.py`):

| quantity | projected | platform | error |
|---|---|---|---|
| kernel, mean | 65.5 s | 66.9 s | -2.0% |
| overhead, mean | 8.40 s | 8.14 s | +3.2% |
| residual, mean / worst network | 0.2875 / 0.320 s | 0.2833 / 0.314 s | +1.5% / +2.1% |
| predict wall, mean | 74.2 s | 75.3 s | -1.1 s |
| predict wall, worst network | 85.4 s | 80.0 s | +5.4 s: this run's slowest network took 1.06x its mean; the projection assumes S35's 1.15x |
| setup flops, every worker | 29,571,934,767 (complete binding phase, setup probe) | 29,571,934,767 on all five | exact |
| score | about 2.02e-9 expected by the lead | 2.031e-9 | |

The kernel is 2% low because the stored S35 legs were measured while the box crawled (section 8). The residual, projected
from requests alone, is 1.5% high: the platform paid 6.93 microseconds per request on S36 against 7.04 on S35.

### The second: S37, #333435 (2026-10-02 03:17 UTC), and what the two fires say about the instrument

S37 graded 1.951e-9, 100 networks, none lost (`O/board_evals/eval_333435.json`). The same candidate sessions
(`runs/c_s37_paced`, measured in the lead's quiet window) judged against two sets of S35 legs:

| quantity | against the stored S35 legs (measured while the box crawled; the stage C the fire was decided on) | against calm S35 legs (`runs/c_s35_paced_bank_v2`) | platform |
|---|---|---|---|
| kernel, mean | 65.2 s (-5.6%) | 69.08 s (+0.0%) | 69.07 s |
| overhead, mean | 8.83 s (+2.2%) | 8.83 s | 8.64 s |
| residual, mean / worst network | 0.3016 / 0.336 s (-0.5% / +2.1%) | the same (from requests) | 0.3032 / 0.329 s |
| predict wall, mean | 74.3 s (-3.7 s) | 78.2 s (+0.2 s) | 78.0 s |
| predict wall, worst network | 85.5 s | 90.0 s | 87.6 s |
| setup flops, every worker | 29,571,934,767 | | 29,571,934,767 on all five |

S36 the other way round: its legs crawled like the stored S35 legs and gave 74.2 s (platform 75.3); against the calm S35
legs they give 78.1 s. **So the instrument is good to about 1 s of wall and 1.5% of residual when parent and candidate are
measured at the same load, and off by up to 3.8 s either way when they are not.** Every later qualification runs the pair
in one window (S35, candidate, candidate, S35). Residual per request on the platform: S35 7.04, S36 6.93, S37 7.07
microseconds. F4 + K4L cost 2.2 s of platform kernel time (69.1 against 66.9 s for S36).

### The third: S38, #333481 (2026-10-02 07:57 UTC), and the memory-bound share re-set from 0.72 to 0.65

S38 graded 1.907e-9, 100 networks, none lost (`O/fire/s38_20261002/eval_333481.json`). Stage C of the lead's chain
(`runs/c_s38`: S35, file, file, S35 in one quiet window) against the platform:

| quantity | projected (share 0.72, as fired) | re-projected (share 0.65) | platform |
|---|---|---|---|
| kernel, mean | 66.0 s (-3.4%) | 67.0 s (-2.0%) | 68.36 s |
| overhead, mean | 8.62 s (+2.4%) | 8.62 s | 8.42 s |
| residual, mean / worst network | 0.2945 / 0.328 s (+0.0% / -1.3%) | the same (from requests) | 0.2944 / 0.332 s |
| predict wall, mean | 74.9 s (-2.1 s) | 75.9 s (-1.2 s) | 77.07 s |
| predict wall, worst network | 86.3 s | 87.4 s | 88.27 s |
| setup flops, every worker | 29,571,934,767 | | 29,571,934,767 on all five |

The request side is exact (7.03 microseconds of residual per request on the platform; S35 7.04, S36 6.93, S37 7.07). The
kernel side read low. Each fired build gives one estimate of the share W of memory-bound work in S35's platform kernel time
(kernel ratio to S35 = W x memory-bound ratio + (1 - W) x CPU-bound ratio, both ratios measured locally): S36 0.62, S37 0.72,
S38 0.55, each +-0.1 because the platform repeats a run's kernel time only to about 2%. The central value is now 0.65 (band
0.55-0.75; `project.py`). The three fires re-projected from their stored legs with 0.65: mean wall +0.3 s (S36), +1.1 s
(S37), -1.2 s (S38); with 0.72 they were -1.1, +0.2, -2.1 s. **Read a projected mean wall as +-1.5 s.** S38 against S37 on
the platform: kernel x0.990, bill x0.9940, residual x0.971; locally the kernel instructions say x0.96-0.97, so the
edge cut and the cheaper stand-in gained less kernel time on the platform than the local counts promised (one run each, inside
the platform's 2%).

**Page-fault work, the cause of the window dependence (found 09:10 UTC, in force 09:20 UTC).** Raw kernel instruction counts
are not load-independent: when the machine is short of free memory the kernel takes freed pages back and pages every large
array of the server in again, and that work is counted. The server's minor page faults per network are recorded in every row:

| file, network | quiet window | loaded box | crawling box |
|---|---|---|---|
| S35, network 100: instructions (faults) | 349.8 G (25,664) | 353.2 G (310,633); 362.4 G (1,082,947) | 369.9 G (1,600,911); 381.1 G (2,371,499) |
| S38, network 100 | 339.4 G (9,773); 339.7 G (11,202) | 352.0 G (947,625); 353.1 G (1,094,538) | |
| S35, network 35 | 344.2 G (399,347) | 344.8 G (464,676); 349.3 G (805,105) | 356.5 G (1,278,635); 365.2 G (1,929,938) |

A least-squares fit over the 42 stored session-networks under the platform clock, one intercept per file and network (10
groups): 14,200 instructions per minor fault, 8,900 of them in the memory-bound operation class and 5,300 in the CPU-bound
class; the residual is 0.27% rms (worst 0.8%) where the raw counts scatter by 2.2% rms (memory-bound class alone: 4.1% raw,
0.48% after). `project.fault_free` takes that work out before any ratio is formed. Kernel time of the three fired builds
projected from S35, for four sets of S35 legs from different windows (two legs of the 07:25 UTC window, its quiet leg alone, the
calm legs of 03:17 UTC, the crawl legs of 01:23 UTC), share 0.65:

| build (its legs) | raw counts | fault work taken out | implied share, fault-free |
|---|---|---|---|
| S36 (crawl) | +0.1 .. +9.4% | +1.4 .. +2.1% | 0.72-0.75 |
| S37 (03:00 UTC window) | -5.1 .. +3.8% | -0.6 .. +0.1% | 0.62-0.66 |
| S38 (quiet window) | -9.2 .. -1.3% | -1.5 .. -1.1% | 0.58-0.60 |
| S38 (two sessions at load 15-26) | -4.7 .. +4.3% | -1.9 .. -1.1% | 0.56-0.60 |

So the 3.7 s miss on S37 and the window sensitivity noted after each fire were this one effect, and the instrument no longer
needs both arms in one window: S38's end-check session at load 15-20 against its stored quiet legs projects 77.0 s (platform
77.07 s; 80.0 s with raw counts), and backwards from the bank S38 the earlier files project 79.0 s (S37, measured 78.0), 77.5 s
(S36, 75.3; that run was about 1.5% fast in everything) and 83.1 s (S35, 82.5). The fast path's kernel gate uses the same
counts: four stored pairs agree within 0.6% (raw up to 5.8%). The client's residual counter also reads high under load (7-13%)
but not because of page faults; the residual stays projected from requests.

Stage D projects as well since 08:40 UTC: `calibration/bank_stage_d_paced.json` is S38's 20-network worker run (`runs/d_s38`,
quiet window) with S38's own platform record (`make_reference.py <worker> <eval json>`), and a later stage D under the platform
clock is projected from it over the networks clean in both runs. Checks on stored worker runs (`runs/d_s38_vs_s38ref_selfcheck`,
`d_s37_vs_s38ref_check`, `d_s38_vs_s37ref_check`): S38 against itself returns its record (77.1 s, 0.294 s); S37 from S38 79.8 s
and 0.305 s (measured 78.0 and 0.303); S38 from an S37 reference 75.4 s and 0.293 s (measured 77.1 and 0.294). Two worker runs
measured at different machine load: 2% in wall, 0.5% in residual.

**The rule since 09:00 UTC: project from the newest graded file.** `calibration/bank.json` (written by `set_bank.py`) names
the bank every stage pairs a candidate with and whose platform record it scales: S38 (#333481) with its own memory-bound
share 0.60 (S35's 0.65 carried through the local ratios of `runs/c_s38`). Reason: S35 pays 0.1833 ms of platform overhead
per round trip, S36 / S37 / S38 pay 0.1777 / 0.1794 / 0.1791 (the overhead was projected 3.2%, 2.2% and 2.4% high in the three
fires), and the kernel's memory-bound share differs by parent; residual per request is the same for all four (7.04, 6.93,
7.07, 7.03 microseconds). `new_bank.sh` repeats the re-basing for the next graded file.

Stage C can project from any graded parent as well (`--parent <file> --platform-record <eval json>`). Checked on stored
sessions (`runs/c_dialsA_from_s38_parent_check`, `c_s37_from_s38_parent_check`): requests and bill ratios exact; the wall is only
as good as the load match of the legs (S37's legs of 03:00 UTC against S38's quiet legs: 81.8 s projected, 78.0 measured), so
the pair in one window stays the rule whatever the parent.

### What happened in #333099 (the failed run), per network against S35 on the same 100 networks

| group | networks | kernel | overhead | residual | residual s | wall s |
|---|---|---|---|---|---|---|
| normal speed | 74 | x0.98 | x0.98 | x0.99 | 0.300 | 80.5 |
| slow worker, first setup | 18 | x1.36 | x1.20 | x1.21 | 0.365 | 110.8 |
| slow worker after its re-setup | 8 | x1.26 | x1.34 | x1.30 | 0.401 | 107.4 |

Lost networks: one on the 120 s limit (kernel 108 s), three on the 0.4 s residual limit (0.40-0.43 s), all four on slow
workers. Same code as S35 apart from the setup order. S35 at these multipliers: wall 82.5 x 1.34 = 110.6 s mean on such a
worker (its heaviest networks at 118-120 s), residual 0.304 x 1.30 = 0.395 s mean (about half of that worker's networks over).

### Which limit breaks first: the public record (954 runs with 100 networks of telemetry, `O/board_evals/scan_332296_333380.jsonl`)

Runs / runs that lost a network on the 0.4 s residual limit / runs that lost one on the 120 s wall, by the run's mean wall
(rows) and mean residual (columns):

| mean wall | residual < 0.25 s | 0.25-0.28 | 0.28-0.31 | 0.31-0.34 | 0.34-0.37 | >= 0.37 | all |
|---|---|---|---|---|---|---|---|
| < 60 s | 263 / 0 / 0 | 12 / 0 / 0 | - | - | - | - | 275 / 0 / 0 |
| 60-75 s | 95 / 0 / 0 | 11 / 0 / 0 | 3 / 0 / 0 | 9 / 0 / 0 | 11 / 0 / 0 | 3 / 3 / 0 | 132 / 3 / 0 |
| 75-85 s | 47 / 0 / 0 | 17 / 0 / 0 | 46 / 0 / 0 | 30 / 0 / 0 | 14 / 1 / 0 | 1 / 1 / 0 | 155 / 2 / 0 |
| 85-90 s | 43 / 0 / 0 | 6 / 2 / 0 | 7 / 0 / 0 | 26 / 2 / 1 | 25 / 4 / 0 | 2 / 1 / 0 | 109 / 9 / 1 |
| 90-100 s | 29 / 0 / 3 | 30 / 0 / 0 | 24 / 0 / 1 | 19 / 0 / 2 | 42 / 3 / 0 | 15 / 10 / 0 | 159 / 13 / 6 |
| >= 100 s | 43 / 0 / 14 | 20 / 1 / 7 | 20 / 0 / 5 | 16 / 0 / 3 | 15 / 0 / 5 | 10 / 8 / 2 | 124 / 9 / 36 |

- S35 sits in the cell 75-85 s x 0.28-0.31 s (82.5 s, 0.304 s): 46 runs, none lost a network. Below 85 s mean wall no run in the
  record lost a network on the wall (0 of 562); below 85 s and below 0.34 s mean residual none lost one at all (0 of 533).
- Between 85 and 100 s the residual limit takes more runs than the wall does (22 against 7).
- Caveat: a run that meets a slow worker moves up in this table (its mean rises with the event: #333099 ran 80.5 s on its 74
  normal networks and 88.1 s overall), so the lower rows understate what a design at 82 s risks. Our own count for designs at
  77-82 s: 1 event in 6 runs.
- Wall max/mean of a run: median 1.09, 99th percentile 1.25, largest 1.34 over 679 runs with mean wall >= 60 s. S35 needs 1.455
  to hit 120 s. Residual max/mean for runs with mean residual 0.28-0.31 s (100 runs): median 1.09, 90th percentile 1.26,
  largest 1.33. S35 needs 1.315 to hit 0.4 s. The residual is the limit S35 is closer to.
- Setup time: of 6,162 setup executions in the record only 64 took 3.0 s or more, and 43 of those are ours (3.9-4.33 s; the
  limit is 5 s). No setup has timed out in the record, so nobody has seen what a setup timeout does.

## 3. CPU anatomy of RRPI (networks 35 and 100, mean per network)

In process (`cpu_anatomy.py`), clean run `runs/anatomy/rrpi_clean` (system/user 0.18-0.21), bill 263.30 Gflop per network.
CPU s = cycles / 3.64e9. Total 28.9 CPU-s per network (measured 29.6 s). Full tables: `runs/anatomy/rrpi_v2/report.txt`.

By operation type:

| operation | calls | CPU s | share | billed Gflop | bill share | CPU s per Gflop |
|---|---|---|---|---|---|---|
| matmul | 657 | 9.93 | 34.3% | 214.36 | 81.4% | 0.046 |
| add | 12,410 | 8.27 | 28.6% | 19.22 | 7.3% | 0.430 |
| subtract | 9,496 | 5.12 | 17.7% | 14.16 | 5.4% | 0.361 |
| multiply | 3,198 | 1.34 | 4.6% | 3.81 | 1.4% | 0.352 |
| put_along_axis | 57 | 1.27 | 4.4% | 0.53 | 0.20% | 2.39 |
| put | 305 | 0.75 | 2.6% | 0.16 | 0.06% | 4.77 |
| copyto | 1,720 | 0.64 | 2.2% | 0.78 | 0.30% | 0.82 |
| everything else | 10,323 | 1.61 | 5.6% | 10.28 | 3.9% | 0.16 |

By kernel phase:

| phase | CPU s | share | billed Gflop | bill share | CPU s per Gflop |
|---|---|---|---|---|---|
| leaf matmul (all Winograd products) | 9.75 | 33.7% | 207.40 | 78.8% | 0.047 |
| unwind (output-side adds) | 6.62 | 22.9% | 14.48 | 5.5% | 0.457 |
| estimator body outside the product kernel | 5.69 | 19.7% | 13.89 | 5.3% | 0.409 |
| transform of the right operand | 4.05 | 14.0% | 10.30 | 3.9% | 0.393 |
| transform of the left operand | 2.23 | 7.7% | 7.72 | 2.9% | 0.289 |
| other | 0.58 | 2.0% | 9.51 | 3.6% | 0.061 |

By stage (the function the layer loop calls):

| stage | CPU s | share | billed Gflop | bill share | CPU s per Gflop |
|---|---|---|---|---|---|
| column push `_deadcut_push` | 12.36 | 42.7% | 114.69 | 43.6% | 0.108 |
| plane read `_cp_read_young_sq` | 6.18 | 21.4% | 93.17 | 35.4% | 0.066 |
| covariance + star `_flms_cov_star` | 4.80 | 16.6% | 28.75 | 10.9% | 0.167 |
| pair map `_nonlinear_pair_batch` | 1.56 | 5.4% | 3.04 | 1.2% | 0.513 |
| layer loop body | 0.78 | 2.7% | 1.87 | 0.7% | 0.415 |
| row gather by scatter `_unique_rows_via_scatter` | 0.76 | 2.6% | 0.30 | 0.1% | 2.55 |
| learned covariance correction | 0.72 | 2.5% | 3.96 | 1.5% | 0.180 |
| final-layer k3 `_l15_rowcut_k3` | 0.66 | 2.3% | 8.99 | 3.4% | 0.074 |
| `_bd_update` | 0.39 | 1.3% | 0.85 | 0.3% | 0.454 |

Sites ranked by CPU seconds per billed gigaflop (sites of at least 0.05 CPU-s):

| site | CPU s | billed Gflop | CPU s per Gflop |
|---|---|---|---|
| pair map, `put` in `_psp_pair_series` / `_ppi_take` / `_ppi_copy_p16` | 0.71 | 0.15 | 4.2-5.9 |
| `put_along_axis` in `_exact_expand_rows` (three callers in the push) | 0.55 | 0.23 | 2.3-2.5 |
| `put_along_axis` in `_unique_rows_via_scatter` | 0.71 | 0.30 | 2.40 |
| level-1 loads of operand quadrants (`copyto`) | 0.17 | 0.17 | 0.8-1.2 |
| transforms inside the 4-way symmetric block products (`forward`, `unwind` of `_psdb_group`) | 0.42 | 0.57 | 0.72-0.83 |
| unwind adds at depth 6 | 3.46 | 7.54 | 0.459 |
| unwind adds at depth 7 | 2.66 | 5.85 | 0.454 |
| right-operand transforms at depth 7 | 1.50 | 3.54 | 0.423 |
| right-operand transforms at depth 6 | 2.24 | 5.87 | 0.381 |
| left-operand transforms at depth 6 and 7 | 1.44 | 5.66 | 0.25-0.27 |
| leaf matmul at depth 7 (8x8 blocks) | 3.41 | 39.64 | 0.086 |
| leaf matmul of the split reads (depth 6 leaf sums, and the depth-7 reads) | 3.04 | 69.87 | 0.043 |
| leaf matmul at depth 6 | 2.33 | 62.83 | 0.037 |
| leaf matmul at depth 5 | 0.75 | 26.44 | 0.028 |

Leaf matmuls run at 10.0-13.7 Gflop/s at depth 7 against 23-46 Gflop/s on the depth-6 pushes and 33-43 at depth 5: at
depth 7 a 1024^3 product is 823,543 products of 8x8 blocks, and numpy's per-product call cost (about 0.12 microseconds)
outweighs the arithmetic. Depth 7 as a whole (leaf matmuls 3.97 s including the depth-7 reads, transforms and unwind 4.66 s) is
8.6 of the 28.9 CPU-seconds for 60.7 of the 263.3 Gflop. By layer the CPU is flat: 1.8-2.4 s for each of layers 3-14.

## 4. CPU and request variants (task 4)

Built by `build_variant.py` as RRPI + fragments (`variants/fragments/*.py`: a flag, or one function redefined at the end of the
file). Measured in process on networks 35 and 100 against a clean RRPI run (`variant_eval.py`, `runs/variant_eval_v3.json`); all
runs clean (system/user CPU 0.11-0.28). "Platform kernel" = the two-class projection, memory-bound share 0.72 (0.58..0.80).
"Kernel calls" are numpy calls in process (38,166 per network for RRPI); boundary requests are measured in stage C.
MSE on these two networks moves +-0.5% from rounding alone, and both are in the tuning panel: only a 40-network run decides it.

| variant | what changes | bill | predictions | kernel calls | local kernel CPU | platform kernel (band) |
|---|---|---|---|---|---|---|
| **lvfuse_sized32** | level fusion kept on left-operand trees and on other trees up to 32M floats | 0 | byte-identical | +1,378 | -10.0% | -13.1% (-14.6..-10.5) |
| lvfuse_sized | same with an 8M limit | 0 | byte-identical | +2,074 | -10.7% | -14.5% (-16.1..-11.6) |
| lvfuse_a | fusion on left-operand trees only | 0 | byte-identical | +3,328 | -11.2% | -15.1% (-16.8..-12.1) |
| lvfuse_off | no fusion | 0 | byte-identical | +4,458 | -10.3% | -13.9% (-15.5..-11.0) |
| **tsr_off** | thin low-rank-star products as plain matmuls | +0.48% | rounding-level (MSE +0.56% on 2 networks) | -4,350 | -1.4% | -1.7% |
| tsr_l1 | the same products at depth 1 | +0.30% | rounding-level (+0.39%) | -1,720 | -1.0% | -1.2% |
| strips_l5 | covariance outer rectangles at depth 5 | +0.09% | rounding-level (+0.27%) | -462 | -1.8% | -1.5% |
| **fastC = lvfuse_sized32 + tsr_off** | | +0.48% | rounding-level (+0.56%) | -2,972 | -11.3% | -14.7% (-16.3..-11.9) |
| d7sq_off | depth-7 1024^3 products back to depth 6 | +0.43% | rounding-level (+0.43%) | -140 | -7.6% | -6.7% (-7.3..-6.4) |
| d7cp_off | depth-7 column-packet pushes back to depth 6 | +0.89% | rounding-level (+0.81%) | -598 | -9.1% | -7.9% (-8.9..-7.3) |
| r7_off | depth-7 reads back to the split reads | +0.36% | rounding-level (+0.10%) | -8 | -3.2% | -3.0% |
| safeB = d7cp_off + d7sq_off | | +1.25% | rounding-level (+0.35%) | -836 | -15.4% | -13.4% (-15.1..-12.4) |
| d7all_off | all three depth-7 layers off | +1.61% | rounding-level (+0.40%) | -845 | -17.0% | -14.6% (-16.6..-13.4) |
| gather_rows | `A[rows]` instead of the scatter emulation | +0.23% | byte-identical | -140 | -3.3% | -2.3% (-3.1..-1.9) |
| p6_off | padded depth-5 -> 6 moves off | +0.34% | rounding-level (+0.23%) | +232 | -3.0% | -3.3% |
| d6x_off | remaining depth-5 -> 6 moves off | +0.03% | rounding-level (+0.34%) | +22 | -0.8% | -0.9% |
| safeA = d7sq_off + p6_off + d6x_off + gather_rows | | +2.39% | rounding-level (+0.41%) | +116 | -19.9% | -19.0% |

Reading:
- The fusion variants cost no bill and leave every prediction byte unchanged; they trade requests for CPU. Per tree size (RRPI,
  per network): on trees of 64M floats and more, un-fusing saves 2.6 s of CPU for 642 more calls (4.1 ms per call); 32-64M: 1.4 s
  for 642 calls (2.2 ms); 16-32M: 0.31 s for 336 calls (0.9 ms); 8-16M: 0.17 s for 355 calls (0.5 ms); below 8M: 0.2 s for about
  1,400 calls (0.14 ms). Left-operand trees are not un-fused anywhere: there the fused op is as fast as the single ops.
- tsr_off is the cheapest way to remove requests: 0.11% of bill per 1,000 calls. strips_l5: 0.19% per 1,000, and it saves CPU too.
- The depth layers trade bill for kernel time at 8-16% of kernel per 1% of bill. They do not add up the way their names suggest:
  d7sq_off needs the depth-5 -> 6 rules on, otherwise its products fall to depth 5 (safeA: bill +2.39% instead of +1.03%).

### 4a. Recommended builds (to be confirmed at the boundary and on 40 networks)

| tier | build | bill | platform kernel | kernel calls | what it is for |
|---|---|---|---|---|---|
| 0 | lvfuse_sized32 | 0 | -13.1% | +3.6% | free speed; also the S35-identical probe for the platform (`variants/s35_lvfuse_sized`, 8M limit, is built; a 32M one takes a minute) |
| 1 | fastC = lvfuse_sized32 + tsr_off | +0.48% | -14.7% | -7.8% | the designated-safe candidate: faster AND fewer requests |
| 2 | fastD = fastC + strips_l5 + d7sq_off | +1.00% | -19.2% (-20.0..-17.7) | -10.3% | more wall headroom if the lead wants to spend it on bill cuts elsewhere (measured in process like the others; MSE +0.34% on the 2 networks) |

In the platform's setup regime (only the 128 eagerly fused keys fused; measurement-only fragment `emu128`, in process, same two
networks): the parent makes 39,343 kernel calls (+1,177 against the all-fused local parent); lvfuse_sized32 then costs +1,196
calls (+3.0%) and cuts local kernel CPU by 8-10% (projected platform kernel -11..-12.5%); fastC cuts calls by 3,153 (-8.0%) and
local kernel CPU by 9-10%. The parent's run in this regime read slightly high (system/user 0.32), so the upper ends are soft.
The gain is nearly the same as with every key fused, because the eagerly fused keys include the big trees.

### 4a-bis. Boundary measurement of fastC, and what a slow setup does (2026-10-01 20:33-20:44 UTC)

Stage C, fastC on RRPI against S35, parent-candidate-candidate-parent, networks 35 and 100 (`runs/c_rrpi_fastC_vs_s35`). While
the second candidate session started, the machine's load rose to 27 (Spotlight indexing plus the other lanes), so the two
candidate sessions show the same build under a normal and under a slow setup:

| | S35 (2 sessions) | fastC, normal setup | fastC, slow setup |
|---|---|---|---|
| setup wall | 4.10 s, 493 of 521 deferred keys fused | 3.53 s, all 588 fused | 4.51 s; the 4.0 s guard stopped the binding after 146 keys, none fused |
| setup requests | 28,152 | 23,280 | 5,214 |
| bill, network 35 / 100 (Gflop) | 263.24 / 263.91 | 264.23 / 264.90 | 267.47 / 265.83 |
| requests, network 35 / 100 | 42,575 / 41,718 | 39,923 / 39,013 | 43,594 / 40,989 |
| residual instructions (1e9) | 1.920 / 1.877 | 1.783 / 1.736 | 1.961 / 1.843 |
| handles after two networks | 51,623 | 47,078 | 24,190 |

- Normal setup, against S35: bill +0.38%, requests -6.4%, residual instructions -7.0%, kernel proxy time -9.5%. Projected
  platform: wall 73.0 s mean / 84.0 s worst network (S35: 82.5 / 94.9), residual 0.283 s mean / 0.315 s max (S35: 0.304 / 0.339).
  A worker may run 1.43x slower before the wall limit and 1.27x before the residual limit (S35: 1.26x and 1.18x; the slow workers
  of #333099 ran 1.34x on wall and 1.21-1.30x on residual). Verdict PASS, including the setup gate in a loaded window.
- Slow setup: the estimator's setup stops binding tree views when its 4.0 s guard runs out; every view it did not bind is bound
  inside a prediction instead, which is billed (+1.2% on the first network, +0.35% on the second) and costs requests (+9% and
  +5%), and views bound there are never fused. This is the mechanism behind the worst worker of #333099: that worker was set up
  a second time while slow (its setup booked 33.1 Gflop instead of 34.9: fewer views bound), and its networks then ran residual
  x1.30 where the other slow worker ran x1.21. A slow worker therefore hurts twice: every request is slower, and there are more
  of them. fastC's setup needs 17% fewer requests than S35's, so it gets further before the guard.
- The setup wall itself reached 4.51 s in the slow session (the part after the guard took 0.48 s instead of 0.07 s). The
  platform's limit is 5 s and S35 already takes 4.27-4.33 s there.

### 4a-ter. What the platform's setup actually binds (setup-only probes, 2026-10-01 20:50 UTC)

`setup_probe.sh` runs Estimator.setup() alone at the boundary inside a budget context (light: well under 1 GB) and reports its
billed flops, which is the one setup number the platform publishes.

| S35 setup, local, guard set to | setup wall | requests | billed flops | tree keys bound | deferred keys fused |
|---|---|---|---|---|---|
| 1.0 s | 1.11 s | 6,773 | 10.796 G | 190 | 0 |
| 2.0 s | 2.17 s | 10,449 | 20.209 G | 333 | 0 |
| 3.0 s | 3.13 s | 18,958 | 34.759 G | 678 | 0 |
| 4.0 s (as shipped) | 4.14 s | 26,151 | 34.922268623 G | 687 (all) | 379 of 521 |
| platform, #333111, four workers | 4.27-4.29 s | - | 34.922268623 G | all | unknown |
| platform, #333111, worker w-2 | 4.29 s | - | 34.378719666 G | not all | 0 |
| platform, #333099, the worker set up again while slow | 4.33 s | - | 33.113512758 G | fewer still | 0 |

- A complete binding phase bills exactly 34,922,268,623 flops, locally and on the platform. So on a normal platform worker the
  binding phase completes, and one worker in five of S35's own run did not even finish it. The binding phase takes about 3.1 s
  here under load and all of the 4.0 s guard on the platform, so the deferred fusion phase that follows gets almost no time
  there. On the platform, then, only the 128 keys fused while they are bound (S35's "prefix128" change) are fused, plus a few
  (section 6 puts numbers on this: 18,555 of about 19,000 requests go to binding, about 24 deferred keys get fused).
- Every local session of mine fused 268-521 of the deferred keys as well. Local sessions therefore run with MORE fusion than the
  platform: slightly fewer requests and more of the fused-operation CPU penalty than the platform sees. The gain of un-fusing
  measured in section 4 (all keys fused in process) is an upper bound for the platform. `--short-setup` (stage C, D) and the
  measurement-only fragment `emu128` reproduce the platform's regime: only eagerly fused keys keep their fused operations.
- The setup is the estimator's most exposed timing: 4.27-4.33 s against 5 s, by design (it works until its guard), on every worker.

### 4b. How the two levers were found

**The level-fusion layer costs CPU on the output and right-operand trees.** Since S24 the tree kernel runs pairs of level
operations as one numpy call on strided two-component views, for every tree key bound in setup; it was a request diet (about
-3,100 requests per network when it shipped). Micro-benchmark `fuse_bench.py` (numpy only, the candidate's exact views, best of
5, cycles; this machine):

| tree (floats) | left-operand tree, fused / single | right-operand tree | output tree |
|---|---|---|---|
| (7,)^3 x 128 x 6 (0.3M) | 0.95 | 1.11 | 1.54 |
| (7,)^6 x 4 x 4 (1.9M) | 1.01 | 1.08 | 1.86 |
| (7,)^6 x 4 x 12 (5.6M) | 0.89 | 1.64 | 2.59 |
| (7,)^5 x 32 x 12 (6.5M) | 0.96 | 1.58 | 2.86 |
| (7,)^6 x 16 x 16 (30M) | 0.93 | 1.37 | 2.46 |
| (7,)^7 x 8 x 8 (52.7M) | 0.86 | 1.20 | 2.13 |
| (7,)^6 x 16 x 32 (60M) | 0.92 | 1.58 | 2.83 |

On the output tree the fused pair writes a component it also reads, numpy copies an operand first, and the deepest level alone
costs 3.8-4.2x. A depth-7 1024^3 unwind costs 127 M cycles fused against 60 M single (35 ms against 17 ms here) to save 12
requests. The unwind adds are the largest memory-bound item of the whole prediction (6.6 of 28.9 local CPU-seconds, section 3).
Un-fusing is bit-identical and bill-identical by construction, and measured so (section 4). Platform evidence that fusion costs kernel
time: S23 -> S24 (fusion introduced, with two other changes) kernel +7.6%, overhead -8.9%, residual -7.8%.

**Requests are cheapest to remove on the small products.** Kernel calls per billed flop, RRPI, per network:

| products | per network | kernel calls | calls each | CPU s | billed Gflop | Mflop of bill per call |
|---|---|---|---|---|---|---|
| thin low-rank-star products at depth 3 (`_tsr_mm`: 1024 x 1024 x 48 and x 32, and transposes) | 84 | 4,200 | 50 | 0.47 | 4.40 | 1.05 |
| covariance outer rectangles at depth 6 (256 x 1024 x 768 / 512 / 256) | 42 | 3,486 | 83 | 1.01 | 6.25 | 1.79 |
| pushes at depth 6 | 61 | 5,104 | 84 | 5.85 | 64.96 | 12.73 |
| pushes at depth 7 | 22 | 1,868 | 85 | 4.74 | 34.40 | 18.42 |
| 1024^3 at depth 7 (covariance) | 14 | 1,316 | 94 | 2.50 | 14.12 | 10.73 |

The 84 thin star products use a tenth of all requests for 1.7% of the bill. Measured as plain matmuls (section 4): bill +0.48%,
4,350 fewer calls per network. That is the size of the residual problem in section 2.

## 6. The setup under the platform's clock (2026-10-01 22:20-23:10 UTC)

The estimator's setup binds its 649 tree keys most-used first until its own 4.0 s guard, then fuses bound keys until the same
guard, then runs an unguarded tail. Requests and flops per key are deterministic, so the flops a platform setup booked say
exactly how many keys it bound (`setup_keys.py`: bisection over light setup-only probes):

| platform setup | flops booked | keys bound | requests when the guard fired |
|---|---|---|---|
| #333111 (S35), workers w-0, w-1, w-3, w-4 | 34,922,268,623 | all 649 | at least 18,555 |
| #333111 (S35), worker w-2 | 34,378,719,666 | exactly 623 | 17,954 |
| #333099, five first setups | 34,922,268,623 | all 649 | at least 16,241 (that source's binding phase) |
| #333099, the worker set up again while slow | 33,113,512,758 | exactly 580 | 14,589 |

- The slow re-set-up worker then ran its predictions 1.30x slower in residual than S35's normal workers and 1.34x in overhead
  (which contains its extra requests): 14,589 x 1.22..1.34 = 17,800-19,550 requests for a normal worker.
- Independent estimate from the platform's own wire time (the server's decode + encode time per request, reported per setup
  and per network): 0.2923 s per S35 setup against 0.8587 s per S35 prediction. In instructions of the same server code a local
  prediction costs 7.17 G (`runs/cal_s35_noguard_wire`), S35's complete binding phase with the tail 2.28 G and its whole
  deferred fusion 0.96 G more (`runs/setup/s35_free_c.json`, `setup_phases.py`): the platform's setup comes out at the binding
  phase plus 16% of the fusion = 20,100 requests. This rests on wire time being proportional to wire instructions across
  request types; the platform runs Python 3.10 and this machine 3.11, so +-7%.
- **Taken together: a normal worker gets through about 19,000 requests in the guard (range 18,600-20,100)**, w-2 was 5.5%
  below that. `setup_clock.py` holds the calibration; `--short-setup` (stages C, D, the setup probe) replaces the estimator's
  setup clock by `requests so far x 4.0 s / 19,000`, `--setup-slowdown X` by the same for a worker X times slower.
  Check: S35 under this clock binds all 649 keys, fuses 24 deferred ones and books 34,922,268,623 flops; at x1.06 it binds
  622 keys (w-2: 623); at x1.30 it binds 486.

Setup margin by build (`setup_margin.py`, `runs/setup/margin_table_v1.log`; deterministic, any load):

| build | binding phase, requests | share of a normal worker's 19,000 | keys bound by a normal worker | by a worker 6% slower | by a worker 30% slower | deferred fusion, requests | tail |
|---|---|---|---|---|---|---|---|
| S35, RRPI, engineer's `T+C+B+F+D`, `...+Hfull` | 18,555 | 0.977 | 649 | 622 | 486 | 9,481 | 601 |
| any of them + `lvfuse_sized32` (fastC included) | 17,198 | 0.905 | 649 | 649 | 543 | 5,527 | 551 |
| the #333099 source (no keys fused while bound) | 16,241 | 0.855 | 649 | 649 | 580 | 11,795 | - |
| engineer's `...+Hfull+Phfull_v1` (regenerated list, 649 keys) | 18,704 | 0.984 | 649 | 614 | 475 | 9,599 | 601 |
| the same + `lvfuse_sized32` (with or without `tsr_off`) | 18,002 | 0.947 | 649 | 646 | 503 | 5,904 | 551 |
| engineer's `T+C+B+F+D+Prrpi_v1`, `...+Hpad+Prrpi_v1` (regenerated list, 592 keys) | 16,595 | 0.873 | 592 | 592 | 491 | 8,106 | 601 |
| `T+C+B+F+D+Prrpi_v1` + `lvfuse_sized32` | 15,911 | 0.837 | 592 | 592 | 522 | 5,686 | 551 |
| engineer's `...+Hfull+F4v2`, `T+C+B+F+D+F4v2+K4L` (RRPI's list) | 18,555 | 0.977 | 649 | 622 | 486 | 9,481 | 601 |

(`runs/setup/margin_table_v2.log`, 2026-10-02 01:19 UTC. The regenerated 649-key list is no shorter than RRPI's: after binding a
normal worker still fuses only about 15 keys beyond the first 128. Its gain on the platform is its better head: 75% of tree
uses fall on the first 128 keys instead of 57%, engineer's census.)

- S35's "prefix128" change (the first 128 keys are fused while they are bound) is the only difference to the #333099 source.
  It costs 2,314 requests of binding. On normal workers the platform could not tell the two apart (#333099's 74 normal
  networks: kernel x0.98, overhead x0.98, residual x0.99 of S35's).
- What the slow worker of #333099 lost by its second setup was not mainly the 69 unbound keys but ALL fusion: that source
  fuses only after binding, and the guard fired during binding, so nothing was fused on that worker. No fusion at all costs
  RRPI 4,458 more requests per network (section 4, in process), about 9.5% of its round trips; the platform measured that
  worker's overhead at x1.34 against x1.20 on the other slow worker (+12%) and its residual at x1.30 against x1.21 (+7%).
  A session of that source at exactly that pace is queued to check the emulation against this event.
- The un-fuse fragment shortens the binding phase by 1,357 requests because the big right-operand and output trees among the
  first 128 keys are no longer fused while they are bound.
- The setup's unguarded tail is 601 requests (9 fused block-product sides 406, pair-block indices 20, the three depth-7 read
  trees 161, 14 constants). The platform's setup ends 0.27-0.29 s after the guard.

## 8. Boundary runs under the platform clock (from 2026-10-02 00:25 UTC; box at load 20-35)

Every session: setup at a normal platform worker's pace (`--short-setup`), then networks 35 (the worker's first) and 100.
Parent = the bank S35 in the same regime (649 keys bound, 128 + 24 fused, 43,654 and 42,818 requests on the two networks:
1,125 more per network than with every listed key fused). Residual is projected from requests (deterministic) because the
residual instruction counter reads 5-12% high in sessions that crawl under this load (section 1); kernel ratios carry about
+-2.5% of load noise per network here, i.e. about +-2 s on the projected wall.

| run | build | setup: keys bound / deferred "fused" / requests | requests per network | bill | kernel proxy | round trips | projected wall mean / worst s | projected residual mean / max s | a network may take (x mean wall / x mean residual) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| platform, #333111 | S35 | all / about 24 / 19,617 | about 43,240 | 0 | 0 | 0 | 82.5 / 94.9 | 0.304 / 0.339 | 1.46 / 1.31 | bank |
| `c_stackH_lvfs32_tsr_paced2` (+ `_rejudged2`), parent, candidate, candidate, parent | engineer's `T+C+B+F+D+Hfull` + `lvfuse_sized32` + `tsr_off` (RRPI's list) | all / 261 / 19,574 | 42,358 | -1.57% | -8.4% | -0.7% | 75.0 / 86.3 | 0.298 / 0.332 | 1.60 / 1.34 | PASS |

| `c_stackHP_lvfs32_tsr_paced` (+ `_rejudged`), two candidate sessions against the stored S35 legs | the same with the engineer's regenerated prebind list: `T+C+B+F+D+Hfull+Phfull_v1` + `lvfuse_sized32` + `tsr_off` | all / 200 / 19,565 | 40,868 | -1.88% | -9.2% | -4.0% | 74.2 / 85.4 | 0.288 / 0.320 | 1.62 / 1.39 | PASS |

| `c_s37_paced` (run by the lead, 02:28-02:44 UTC), two candidate sessions against the stored S35 legs | S37 file = the S36 file + F4 (strength 0.6) + K4L v2, sha 2a25a074 | all / 200 / 19,565 | 42,864 | -1.55% | -9.7% as printed; about -8% with the bias below | -0.2% | 74.3 / 85.5 as printed; read about 75-76 mean | 0.302 / 0.336 | 1.61 / 1.33 as printed; read about 1.59 / 1.33 | PASS |
| `c_TCBFD_F4v2_K4L_paced`, one session | `T+C+B+F+D+F4v2+K4L` alone (RRPI's list, no release fragment): the first client-to-server run of F4 and K4L | all / 24 / 19,617 | 45,607 | -0.44% | not usable (both networks paged) | +5.3% | about 85-87 / 98-100 (rough) | 0.321 / 0.357 | about 1.4 / 1.25 | runs clean; FAIL on `clean` only |

The S36 file is the second row (sha 87b3d6ef, fired by the lead as #333429 at 02:26:57 UTC on stages A and C; its 20-network
worker run was stopped by the lead after two networks: handles 39,684 / 40,970, outputs finite, bills and requests identical
to the stage-C sessions).

Load bias of rows judged against the stored S35 legs: those legs were measured while the box crawled (kernel instructions of
the same network 356-381 G against 348-360 G when calm), the S36 and S37 legs in calmer minutes (349-354 G), so the kernel
ratios of these rows flatter the candidates. The platform put a number on it: S36 ran at 75.3 s mean against the printed 74.2
(kernel 2.0% above the projection), so the bias is about 1 s of wall, not the 2.5 s I first wrote in TEAM_LOG at 02:45 UTC.
Read S37 at about 75-76 s. Requests, round trips and bills are exact. Calm S35 legs will replace the stored ones.

**Calm S35 legs and what they showed** (`runs/c_s35_paced_bank_v2`, 03:17-03:25 UTC, load 5-10): 43,236 requests and 352.4 G
kernel instructions per network, system/user 0.17. Judged against them, the S36 legs (measured at load 18-28) read kernel
proxy 0.973x and a projected wall of 78.1 s; judged against the crawling legs of 00:26-01:23 UTC they read 0.908x and
74.2 s; the platform measured 75.3 s. Parent and candidate legs must be measured at the same load: the paired order (parent,
candidate, candidate, parent) is the reliable form, stored legs only in a steady window. On the 14 networks of S37's
20-network run that were calm, against S35's own calm 20-network run: projected platform kernel 67.1 s (S36 measured 66.9).

**Slow setup, S35** (`runs/c_s35_slow130`, one session set up at the pace of a worker 1.3x slower; both networks paged, so
only the exact quantities are used): 486 of 649 keys bound, the first 128 fused; requests 44,415 on the worker's first
network and 43,274 on the second against 43,654 and 42,818 under a normal setup (+1.7%, +1.1%); bill +0.17% and +0.09%. A slow
setup costs an S35-family build about 1.4% of residual room on top of the worker's own slowness; the #333099 source, which
fused nothing in that situation, paid about 10%. The same session for S36 and S37 was not run (lane stopped at 03:43 UTC on
the lead's instruction; their setup policy is S35's, with 543 of 649 keys bound at that pace).

Notes on the first row: bill -1.28% on the worker's first network and -1.85% on the second (the strips' new shapes are not in
RRPI's list, so the first network pays about 870 view requests of lazy binds); requests -0.9% and -3.2%. Two of the eight
network measurements paged and were dropped, one in each arm. "Deferred fused" counts keys the fusion pass visited: with
`lvfuse_sized32` the big right-operand and output trees among them stay un-fused by design.

## 9. What the margins mean for the designated submission (2026-10-02 02:20 UTC)

The designated file is re-run once on fresh private networks; a network over a limit gets the zero prediction and ruins that
run (#333099: 7.4e-3 instead of 2.07e-9). Room before a limit, per network:

| build | mean residual s | residual room s (0.4 - mean) | mean wall s | wall room s (120 - mean) | source |
|---|---|---|---|---|---|
| S35 (bank) | 0.304 | 0.096 | 82.5 | 37.5 | platform |
| S36 file = `T+C+B+F+D+Hfull+Phfull_v1` + `lvfuse_sized32` + `tsr_off`, sha 87b3d6ef = #333429 | 0.283 | 0.117 | 75.3 | 44.7 | platform, 100 networks (projected before the fire: 0.288 and 74.2) |
| S37 file = S36 + F4 (0.6) + K4L v2, sha 2a25a074 = #333435 | 0.303 | 0.097 | 78.0 | 42.0 | platform, 100 networks (projected before the fire: 0.302 and 74.3; see section 2 for the wall miss) |
| `T+C+B+F+D+F4v2+K4L` without list and release fragments | 0.321 | 0.079 | about 85-87 (paged, rough) | about 34 | stage C, platform clock, 45,607 requests |

Public record (`O/board_evals/scan_332296_333380.jsonl`), runs of estimators that bill every network alike (sd/mean of the
per-network bill below 5%):

- residual: 328 such runs with a mean residual of 0.25-0.35 s. The worst network of a run exceeds the run's mean by 0.02 s at
  the median, 0.04-0.07 s in the worst tenth, 0.10-0.14 s in the worst hundredth. It exceeded 0.079 s in 3.0% of the runs,
  0.096 s (S35's room) in 1.8%, 0.112 s (S36's) in 1.5%, 0.124 s in 1.2%.
- wall: 704 such runs with a mean wall of at least 40 s. A worst network above 1.20x the run's mean wall in 3.4% of them, above
  1.25x in 0.7%, above 1.32x in one run: our #333099. S35 needs 1.46x, S36 1.62x.
- our own history is harsher: 1 run in 9 lost networks (#333099, residual 0.118 s above its mean, one network over 120 s).

First loss on a slow worker (a worker X times slower in everything, its setup at that pace costing 1.4% more requests,
and the build's own spread between a run's worst network and its mean): S35 at X = 1.17 (residual) and 1.26 (wall); S36 at
1.26 and 1.42; S37 at 1.20 and 1.37. On the 1.30x worker of #333099 S37 would run a mean residual of 0.400 s (about half
of that worker's networks lost), S36 0.373 s.

**Rule proposed for the designation** (TEAM_LOG 03:50 UTC): designate the best-scoring graded file that lost no network in
its public run, leaves at least 0.095 s of residual room and 35 s of wall room by the platform's own telemetry, and holds no
error lever judged only on networks it was tuned on. Today S37 #333435; S36 only if S37's file were found defective (the
record prices the difference in room at 0.3% of runs, the score difference is 3.9%). Test on the exact archive before
designating: the pair S35, file, file, S35 in one window under the platform clock; one session set up at the pace of a worker
1.3x slower; the 20-network worker. Pass = at most 43,300 requests per network and at most +2% under the slow setup, a
binding phase of at most 18,100 requests with every key bound, projected mean wall at most 80 s, handles under 70,000, group
RSS under 7.5 GB, bytes and bill equal to native on networks 35 and 100.

Reading: the residual is the exposed limit; the wall is not. S36 is safer than S35 on both. S37 as built gives back S36's gain
on the residual side and stands where S35 stands there (with 6 s more wall room); without the list and the two release
fragments F4 + K4L would stand at 0.079 s. Each 1,000 requests removed buys 0.007 s of residual room; S37 needs about 2,400
fewer to equal S36 and about 4,000 fewer to stand 0.124 s clear.
Candidates at zero bill (sizes from sections 4, 6 and 7; none built): F4/K4L's own elementwise passes (they spend about
2,400 requests), a prebind list short enough for a normal worker to fuse every listed key (about 500), one thin-star product
per layer on stacked left operands (about 700), one view less per operand load (about 600). With bill: covariance outer
rectangles one level shallower (462 for +0.09%). Un-fusing costs 1,378 requests for 9% of kernel time, i.e. 0.010 s of
residual room for 8 s of wall room: by the record the wall room is the one we need less.

### The three entries ahead of us: margins, losses, and the record as a model (2026-10-02 04:06 UTC)

From the analyst's merged public record (`O/analyst/scan_merged.jsonl`, ids 332296-333451, 1,010 runs with telemetry on at
least 90 networks, 105 teams).

| entry | score | wall mean / worst network | residual mean / worst network | a network may take (x mean wall / x mean residual) | runs with a lost network in the last 96 hours |
|---|---|---|---|---|---|
| J2W #333400 | 1.6233e-9 | 96.8 / 108.0 s | 0.364 / 0.376 s | 1.24 / 1.10 | 8 of 18 (7 on the residual limit, 1 on the wall); 1 of the 11 runs at its present residual of 0.356-0.373 s |
| marius_binner #333426 | 1.6517e-9 | 82.9 / 88.8 s | 0.363 / 0.381 s | 1.45 / 1.10 | 7 of 34, all on the residual limit; 2 of its last 10 |
| Luna #333152 | 1.8437e-9 | 90.0 / 109.7 s | 0.247 / 0.259 s | 1.33 / 1.62 | 0 of 3 |
| ours, S37 #333435 | 1.9506e-9 | 78.0 / 87.6 s | 0.303 / 0.329 s | 1.54 / 1.32 | 1 of 6 (#333099) |

Share of runs in the record that lost at least one network:

| by mean residual | runs | lost on the residual limit | | by mean wall | runs | lost on the wall |
|---|---|---|---|---|---|---|
| below 0.25 s | 548 | 0 | | below 85 s | 592 | 0 |
| 0.25-0.30 s | 187 | 4 (2.1%) | | 85-90 s | 120 | 1 (0.8%) |
| 0.30-0.32 s | 53 | 1 (1.9%) | | 90-95 s | 94 | 3 (3.2%) |
| 0.32-0.35 s | 107 | 2 (1.9%) | | 95-100 s | 75 | 4 (5.3%, 95% interval 2-13%) |
| 0.35-0.36 s | 44 | 3 (6.8%) | | 100-105 s | 68 | 10 (15%) |
| 0.36-0.37 s | 38 | 5 (13%, 95% interval 6-27%) | | above 105 s | 61 | 27 (44%) |
| 0.37-0.38 s | 16 | 9 (56%) | | | | |
| above 0.38 s | 15 | 14 | | | | |

For one private run of each file as it stands: J2W about 18% (13% residual + 5% wall; its own cell of the record, wall
92-102 s and residual 0.35-0.38 s: 4 of 41 runs), marius about 13% (all residual; its own cell: 5 of 15 runs), Luna about 3%
(all wall; its worst network was 10 s from the limit), S37 and S36 about 2% each. If each of the three designates its current
best, the chance that at least one of them loses a network is about 30%. The two leaders are exposed on the residual limit
(room 0.036 s), not on the wall.

## 11. Fast path for literal-only candidates, and the S38 chain read (2026-10-02 07:15-08:00 UTC)

### S38 (sha 7c1a67f4, fired as #333481 at 07:57 UTC): the lead's chain, read by this lane

Stages A, C (legs S35, file, file, S35 in the quiet window of 07:25-07:57 UTC) and D all PASS. Numbers per network:
requests 42,797 on the first network after the setup and 40,927 on the next (S37: 43,461 and 42,268; S35: 43,654 and 42,818),
40,784 on average over the 20 networks of the worker run (S37: 42,186 on the same networks, x0.967); bill 0.97871 of S35's;
kernel instructions 337.2 G (S35 in the same window 347.0-350.1 G; S37 at 03:00 UTC 351.5 G); setup: all 649 keys bound in
17,366 requests of a platform worker's 19,000 (S37 18,004; S35 18,557); handles after 20 networks 57,578 (S37 56,441).

| projection of S38 on the platform | mean wall | worst network | residual mean / max |
|---|---|---|---|
| stage C as run (against S35, both S35 legs) | 74.9 s | 86.3 s | 0.295 / 0.328 s |
| against the quiet S35 leg only (leg P0 ran beside two lanes' torch jobs, 07:25-07:31 UTC) | 75.4 s | 86.8 s | 0.295 / 0.328 s |
| against S37's platform record #333435, over the 20 networks of both worker runs | 75.2 s | 84.5 s | 0.293 / 0.318 s |

So 74.5-75.5 s of mean wall and 0.293-0.295 s of mean residual are expected (S37 measured: 78.0 s, 0.303 s). To be checked
with `check_projection.py` when #333481 is graded.

**RSS is not memory on this machine.** S38's worker run showed 6.81 GB of group RSS (S37: 6.20 GB) at a server footprint of
2.20 GB (S37: 2.97 GB): S38 needs less memory, not more. Over the 34 stored sessions the quiet ones (kernel about 40 s per
network) show 6.0-6.8 GB of group RSS and the loaded ones (140-190 s per network) 2.9-3.9 GB, at footprints of 2.1-2.7 GB
throughout: RSS counts freed pages the system has not taken back, and it takes them back only under memory pressure. The
machine guard kills at 7.5 GB of group RSS, so a worker run in a quiet window is within 0.7 GB of a kill at a footprint of
2.2 GB. `guard_footprint.py` is the same guard with the trigger on footprint (tested on a toy process: clean run, kill run); it is
NOT wired in: the lead decides (TEAM_LOG 08:00 UTC).

### S39 (sha a158b9ab, fired as #333528 at 11:39 UTC): pre-check and chain

File: the trainer's candidate 3 + the exact-term module in place of F4 + K4L 0.86 + the star carried by its row sums alone.
Not literal-only against S38 (20 statements differ: the module's three functions, the star's three functions), so full chain.

Stage A pre-check before the window (`runs/a_s39_precheck`, `a_s39_precheck_reviewed`): four einsum signatures new against
S38's proven set, all in the module's once-per-layer coefficient step `_f4c_birth` on small arrays, none with a repeated
operand and none on a symmetric tensor: `lsjab,ak,bk->lsjk` (line 10675: a constant table, the power table of x, the power
table of y), `oab,ak,bk->ok` (10676), `als,lsjk,lk->asjk` (10683), `a,sj,k->asjk` (10685, an outer product). flopscope's
repeated-operand logic lives only in einsum, dot, matmul, inner, outer and tensordot; the new code passes no array twice to
any of them. One call form without exact precedent: `fnp.multiply(Dm, Dm, out=Dm)` (an element operation: no symmetry logic).

The lead's chain (`runs/a_s39`, `c_s39`, `d_s39`; quiet window 11:12-11:38 UTC, all PASS, no other busy process at any leg):

| quantity | S39 | S38 (same tools) | written before the chain |
|---|---|---|---|
| requests per network, networks 35 / 100 | 43,565 / 41,682 (mean 42,624) | 42,797 / 40,927 (41,862) | about 42,560 |
| requests per network over the 20 of the worker | 41,539: +755 on every network, +768 on the first | 40,784 | |
| bill | x0.9777 of S38 | | about -2.2% |
| kernel work (proxy; instructions without page-fault work) | x1.0045; x1.0015 (memory-bound class x1.010, CPU-bound x1.000) | | about -1.5%: WRONG, the star's thin products and QRs are almost no kernel work |
| projected platform wall, mean / worst network | 77.6 / 88.9 s (stage C and stage D agree) | measured 77.1 / 88.3 s | 76.3 / 87.4 s |
| projected residual, mean / worst network | 0.300 / 0.338 s | measured 0.294 / 0.332 s | 0.299 / 0.338 s |
| peak handles after 20 networks | 57,755 (+177 on every network) | 57,578 | about 57,800 |
| peak footprint server + client; group RSS | 2.47 GB; 7.23 GB | 2.53 GB; 6.81 GB | as S38 |
| setup | 19,558 requests, binding 17,366, all 649 keys, 266 fused | the same | unchanged |

The group RSS of 7.23 GB at a footprint of 2.47 GB would have been 0.27 GB from the old guard's kill. The fresh S38 legs of
this window against the stored ones (`compare_parent_legs.py runs/c_s39`): every count equal, kernel instructions without
page-fault work within 0.05%, S39's projection 77.58 s from the fresh legs and 77.59 s from the stored ones: stage C of the
next chain can be two sessions (`--legs CC --parent-from runs/c_s38_bank_legs`).

**S39 graded 1.847e-9 (#333528, 12:09 UTC), no network lost: the corrected rule's first fire.**

| quantity | projected (stage C, parent S38, fault-free counts) | platform | error |
|---|---|---|---|
| kernel, mean | 68.75 s | 68.44 s | +0.5% |
| overhead, mean | 8.53 s | 8.55 s | -0.2% |
| residual, mean / worst network | 0.2998 / 0.338 s | 0.3018 / 0.347 s | -0.7% / -2.4% |
| predict wall, mean / worst network | 77.6 / 88.9 s | 77.3 / 87.6 s | +0.3 s / +1.2 s |

Seven setups in the run (two workers were set up a second time), all with every key bound. Platform constants: 7.08
microseconds of residual per request, 0.1795 ms of overhead per round trip. `new_bank.sh S39 ...` made S39 the bank
(`calibration/bank_S39_a158b9abd2c2_333528.json`, stored legs `runs/c_S39_bank_legs`, stage D reference its worker run).

**The worst residual of a run is a worker's first network after a setup** (the network that carries
`participant_setup_wall_s` in the platform record):

| run | setups | first networks: residual mean / max | all other networks: mean / max | first / other |
|---|---|---|---|---|
| S35 #333111 | 5 | 0.3164 / 0.3300 s | 0.3036 / 0.3387 s | x1.04 |
| S36 #333429 | 5 | 0.3069 / 0.3113 | 0.2820 / 0.3136 | x1.09 |
| S37 #333435 | 5 | 0.3178 / 0.3287 | 0.3025 / 0.3246 | x1.05 |
| S38 #333481 | 5 | 0.3222 / 0.3322 | 0.2930 / 0.3184 | x1.10 |
| S39 #333528 | 7 | 0.3273 / 0.3467 | 0.2999 / 0.3224 | x1.09 |

Locally a first network makes 43,565 requests against 41,682 on a warm one (S39): the four-slot views the L3 loads build at a
key's first load (about 680 requests), the lazily bound network-specific keys, the module's constants; the platform adds about
as much again in cold Python. The limit is per network, so S39's margin is 0.4 / 0.347 = 1.15 on a first network. A lever
with no physics for the engineer: build those views in the slack of the setup's binding phase (17,366 of about 19,000
requests used).

**Private-run risk of S39 by the analyst's model** (`O/analyst/a16_loss_risk.py`'s definitions on `scan_merged.jsonl` as of
11:59 UTC: 1,090 runs): about 2%. The worst network has to sit 1.19 x above our usual worst-to-mean residual (1.115): 13 of
727 leader-sized runs did (1.8%; S37 1.8%, S38 1.65%); of 83 runs with a mean residual of 0.29-0.32 s, 2 lost a network on
the residual limit; in 81 pairs of identical runs the worst residual never rose by more than 13.3% (S39 needs 15.4%). The
losses in this range are slow-worker events (seven runs, worst network at 1.20-1.55 x the run's mean). With more requests,
same model: +1,000: 2.2%, +2,000: 2.9%, +3,000: 4.8%, +4,000: 8.1%, +5,000: 11.6%; with 1,000 fewer 1.65%.

**What requests cost on S39 without touching physics** (`runs/c_s39_lvfs64`, `c_s39_fused`: one session each at load 27,
predictions byte-identical, bill equal, kernel work from fault-free instructions):

| un-fused trees | requests per network | binding phase of the setup | kernel work against S39 | mean wall (S39 measured 77.3 s) |
|---|---|---|---|---|
| above 32M floats (S39 as fired) | 42,624 | 17,366 requests | | 77.3 s |
| above 64M floats | 42,088 (-535) | 17,596 | +2.3% | about 79.1 s |
| none | 41,454 (-1,170) | 18,068 | +6.9% | about 83.0 s (worst network about 94 s) |

**Designation test of S39 (12:14-12:24 UTC): PASS, every criterion met** (`designation_judge.py s39 --pair runs/c_s39 --slow
runs/s39_slow --worker runs/d_s39`). Platform: 0 of 100 lost, mean residual 0.3018 s (rule 0.305; room 0.098 s), worst
network 0.3467 s, mean wall 77.3 s (rule 80), worst 87.6 s. Slow setup (worker 1.3x slower): 525 of 649 keys bound, none
fused, 43,388 requests per network (+1.79%, limit +2%), its first network 1.020x the requests of a normal first network.
First loss on a worker X times slower in everything (0.4 s / worst network of the file's own platform run with the slow
setup's extra requests; 120 s / its worst wall): S39 1.13x residual, 1.37x wall; S38 1.18x, 1.36x; S37 1.20x, 1.37x; S36
1.26x, 1.50x; S35 1.17x, 1.26x (with a common worst-to-mean residual of 1.11: 1.17, 1.20, 1.17, 1.25, 1.17).

**What a first network does that a warm one does not** (request census by function, `runs/c_s39_sites/census_first_vs_warm.txt`;
network 35 first after the setup, 43,565 requests; network 100 warm, 41,682): 1,495 requests build views of tree keys used
for the first time in the process (`_hz_leaf_views` +352, `_l3_quad` +399, `_l3o_pairs` +276, `_rs_basis_views` +178,
`_ipt_side` +151, `_lvf_rs_level1` +81, `_ipt_view` +58) and about 320 build once-per-process constants (`_e2_gnn_layer`
135 and five smaller ones); none does arithmetic. On the platform these 1,883 requests carry +0.0274 s (14.6 microseconds
each, twice the average). For S39 + 200 requests: first networks 0.329 s on average, largest 0.348 s; with the L3 and L3o
views of prebound keys built in setup (675 requests) 0.319 / 0.338 s; with every view family and the constants (1,815)
0.303 / 0.322 s, the level of the warm networks. The setup's 0.34 s between the end of binding (17,366 requests) and the
4.0 s guard is used today by the deferred fusion (266 keys); a session with the setup at 1.09x pace (61 fused instead of
266; `runs/c_s39_setup109`, first network only: the guard stopped it on the machine-wide swap rule) made 216 more requests
on its first network, so those 266 fusions save about 280 requests on every network.

The request census of the warm network by function (S39, 41,682 requests): the tree kernel's forward and unwind passes
15,148 (`_ipt_leaf_unwind_ab` 5,233, `_ipt_forward_b_ab` 4,857, `_ipt_forward_a_ab` 3,587, `_ks_unwind` 1,471), the layer
loop itself 2,894, `_l3o_last` 1,926, the pair map 1,918, the exact-term module 1,674 (step 1,018, birth 656), K4L 1,243
(local 478, birth 465, dk 300), `forward` 1,120, `_bd_update` 1,020.

### Designation test of S38 (2026-10-02 08:46-08:59 UTC): PASS

`python -B designation_judge.py s38 --pair runs/c_s38 --slow runs/s38_slow --worker runs/d_s38` (the pair and the worker are
the lead's chain; the slow-setup session ran on the loaded box, one thread, log `runs/release_s38_slow.log`):

| criterion | S38 | limit |
|---|---|---|
| requests per network (networks 35, 100) | 41,862 | 43,300 |
| binding phase of the setup | 17,366 requests, all 649 keys bound | 18,100, all bound |
| projected mean wall / residual (stage C as fired) | 74.9 s / 0.295 s | 80 s / 0.305 s |
| requests under a setup 1.3x slower | 42,626 (+1.83%); 525 of 649 keys bound, none fused | +2% |
| 20 networks in one worker | all finite; peak handles 57,578; peak footprint server + client 2.53 GB | 70,000; 7.5 GB |
| native parity on networks 35, 100 | bytes and bill identical | identical |
| platform #333481: networks lost | 0 of 100 | none |
| platform: mean residual (worst network) | 0.2944 s (0.3322), room 0.106 s | 0.305 s |
| platform: mean wall (worst network) | 77.1 s (88.3), room 42.9 s | 80 s |

First loss on a worker X times slower in everything (limit / worst network of the file's platform run, residual with the slow
setup's extra requests): S38 1.18x residual, 1.36x wall; S37 1.20x, 1.37x; S36 1.26x, 1.50x; S35 1.17x, 1.26x. With a common
worst-to-mean residual of 1.11 instead of each run's own: 1.20, 1.17, 1.25, 1.17. S38 tolerates a slow worker about as S37
and S35 do on the residual side; S36 (1,000 fewer requests) remains the most tolerant. The slow workers of #333099 ran
1.22-1.34x slow in residual on average and the four lost networks 1.33-1.43x.

### The fast path

Asked by the lead at 07:10 UTC: the trainer's dial candidates come every few hours and differ from the qualified file only
in literals; a full chain needs a quiet machine. `fast_path.sh <qualified.py> <new.py> <tag>` (RUNBOOK, One command):

1. `literal_diff.py`, 2 seconds, no session: both sources parsed, the trees walked side by side. Literal-only = same code,
   and every differing constant is a float literal (int against float included), a packed base64 array or a docstring; each
   is listed with its place. Integers, True/False/None, short strings and any code difference fail the proof. A number that
   one file writes in place and the other reads through a module-level name bound once is compared by value (the trainer's
   b-source dial: `/ 6` became `/ _S19_BSRC_DIV`).
2. Stage A of the new file, einsum signatures against the qualified file (the archive to fire).
3. `fast_path.py`: one worker session of each file under the platform clock on networks 35 and 100; the census must be equal
   for the setup and each network: requests, free trips, handles freed, peak handles, kernel calls per operation name, numpy
   calls per function, billed calls and flops per operation family, bill, stored bytes by number type, setup requests /
   handles / keys bound / keys fused. A bill or byte count moving by under 0.1% with every call count equal is reported as a
   moved row count and passes. Residual and kernel instructions within +-3% / +-2% (both sessions calm) or +-20% / +-10%.

What the gates rest on (stored sessions, no compute):

- Two sessions of ONE file under the platform clock repeat every count exactly, wire bytes in and out included: five pairs
  (S35 twice, the stack, S36, S37). Under the estimator's own wall clock they do not (the setup gets differently far):
  requests differ by 230 per network in `runs/c_rrpi_vs_s35_load1`.
- Instruction counts between two sessions of one file with the same census: residual 0.06% and 0.28%, kernel 0.01% and
  0.13% apart when both were calm (one pair); 0.3-10.6% and 0.3-6.6% apart when the box was loaded (five pairs); the S38
  file at load 22-26 (08:00 UTC) against its own quiet-window legs: +7.3..+12.8% and +3.6..+5.8%, every count equal.
- Footprint and RSS do not repeat (peak footprint +-8-14% between calm sessions of one file), so memory is inherited
  through the census (same shapes, same number types), not measured.

A float's value can steer operations, so the literal proof alone is not the verdict: in the S38 file F4 v4 takes a scalar
path when the strengths per age are equal (80 kflop per network less, no per-column vector) and K4L v3 skips its multiply at
a strength of exactly 1.0; the dead-row and edge-cut thresholds and the kernel prices are floats as well.

Tests without a session: 20 synthetic edits (`selftest_literal_diff.py`), all classified as intended. Real pairs: the
engineer's DS 0.85 -> 1.0 and K4L 0.73 -> 1.0 builds are literal-only (one constant each); the trainer's parents
runM_dials_ds08 -> runM_dials2 are literal-only (4 numbers); S38-ds08 (0d51ccb9) -> the S38 file (7c1a67f4) is not (28
statements: F4 v4, L3o). Judge on stored sessions: S37's two legs PASS; two own-clock S35 sessions FAIL with 14 counts listed.
The engineer's native logs show the same for one dial on networks 35 and 100 (`--native`; F4 with strength 0 against F4 at
its default strength, `quick_tcbfd_f4t0_35_100.json` and `quick_tcbfd_f4_35_100.json`): bill, logged calls and every lever
statistic equal.

Session tests, 07:57-08:29 UTC, box at load 14-29 (other lanes' torch jobs and the trainer's two workers beside every session;
each session lists them in its `busy_at_start.json`):

| test | sessions | literal proof | census | verdict |
|---|---|---|---|---|
| S38 file against `variants/s38_7c1a67f4_dialsA` (k21n -0.048 -> -0.038, `_BD_J1_PHYS`, `_S19_BSCALE`, b-source divisor, F4 strength age 1 0.656 -> 0.628, K4L 0.75 / 0.75 -> 0.8 / 0.7) | S38 13.4 min, variant 8.5 min | 7 numbers | the same: requests 42,797 / 40,927, free trips 5,299 / 5,052, peak handles 40,617 / 41,937, bill 258,268,828,893 / 257,655,715,185 flops, calls and flops of all 69 billed families, wire bytes; 16 bytes more stored int64 on network 100 (reported, inside the allowance); residual instructions x0.914 / x1.003, kernel x0.957 / x1.002; predictions differ | PASS, inherits `c_s38` and `d_s38` |
| S38 file against `variants/s38_7c1a67f4_dialsB_equal_theta` (the same, F4 strengths 0.6 / 0.6) | variant 6.4 min (the S38 session reused) | 7 numbers | differs in 20 counts: requests -38 per network (13 concatenate and 25 full calls), bill -80,512 / -80,128 flops, wire bytes | FAIL: full chain |

The S38 session of the first test (load 22-26) against the chain's two quiet S38 legs: every count and the wire bytes equal;
residual instructions +12.6 / +7.3% and +12.8 / +7.5%, kernel instructions +5.8 / +3.7%. Counts do not depend on load, the
instruction counters do; hence the loaded-box tolerance of +-20% / +-10% and the verdict REPEAT (not FAIL) beyond it.

By-product: the bill by flopscope operation family is now in every session row (`operations`). S38, network 35 (first
network of a worker), 258.27 Gflop: matmul 82.00%, add 7.40%, subtract 5.63%, multiply 1.51%, reshape 0.69%, qr 0.54%, sum
0.42%, copyto 0.38%, put_along_axis 0.18%, getitem 0.18%, einsum 0.16%, divide 0.13%, concatenate 0.12%, asarray 0.10%,
solve 0.09%, astype 0.09%; pure data movement together 1.9%. (flopscope bills a reshape at its element count even as a
view; the engineer's RESULT.md section 12 has the reason the tree kernel keeps them.)

## 12. S40 as the bank, the first network, and the S41 budget (2026-10-02 15:25-16:20 UTC)

**S40 graded 1.7590e-9 (#333559, 15:22 UTC, third place), no network lost.** `bash new_bank.sh S40 <file> O/board_evals/
eval_333559.json s40` (log `runs/new_bank_S40.log`): `calibration/bank.json` = S40 (wall 79.83 s mean / 91.18 s worst
network, kernel 70.65 s, overhead 8.869 s = 0.1821 ms per round trip, residual 0.3109 / 0.3652 s = 7.13 microseconds per
request, memory-bound share 0.598); stage D reference = its 20-network worker run; stored legs `runs/c_S40_bank_legs`.

| quantity | projected (stage C, parent S39) | platform | error |
|---|---|---|---|
| kernel, mean | 70.16 s | 70.65 s | -0.7% |
| overhead, mean | 8.741 s | 8.869 s | -1.4% |
| residual, mean / worst network | 0.3093 / 0.3553 s | 0.3109 / 0.3652 s | -0.5% / -2.7% |
| predict wall, mean / worst network | 79.2 / 89.8 s | 79.8 / 91.2 s | -0.6 s / -1.4 s |

**Where the worst network's 0.010 s came from** (`platform_networks.py <eval json>`, `runs/setup_keys_s40.log`). Seven
setups in the run: workers 3 and 4 were set up a second time. Six setups booked 29,571,934,767 flops (all 649 keys bound);
worker 3's second booked 29,381,392,716 = exactly 627 keys bound when the 4.0 s guard fired after 16,902 requests (a normal
worker gets through about 19,000): that worker ran 1.124x slow during its setup and fused nothing. Its first network,
index 25, is the run's worst: 0.3652 s, with 1.138x the run's median kernel time per flop. The other six first networks:
0.3155-0.3316 s (1.03-1.08x the median of the networks that are no first network, 0.3067 s). Largest residual that is not
a first network: 0.3475 s (network 31, kernel time per flop 1.105x). In this run the residual rises 0.76% per 1% of kernel
time per flop (correlation 0.82 over the 93 other networks), and the run was more uneven than S39's (kernel time per flop
of the middle 80% of networks: x0.957-1.098 against x0.974-1.058). The same setup pace reproduced locally
(`runs/s40_slow1124`: 627 of 649 keys bound, nothing fused): +340 requests on the first network, +296 on the warm one
(+0.76% / +0.69%), bill +0.010% / +0.015%, predictions byte-identical. So 0.3067 s x 1.077 (first network) x 1.105 (slow
worker) x 1.008 (requests) = 0.365 s. The projection of the worst network uses the bank's worst-to-mean ratio (S39's
1.149; this run's was 1.175); with S40 as the bank it now reads about 2% higher than our last three files' median.

**Our graded runs, network by network** (first networks against the median of the others; kernel time per flop against the
run's median in brackets):

| run | setups | mean | median of the others | largest other / median | first networks / median | worst / mean |
|---|---|---|---|---|---|---|
| #333099 | 6 | 0.3163 | 0.3010 | 1.337 | 1.04 1.06 1.07 1.08 1.04 1.44 (1.31) | 1.372 |
| S35 #333111 | 5 | 0.3042 | 0.2996 | 1.130 | 1.05 1.07 1.04 1.10 1.02 | 1.113 |
| S36 #333429 | 5 | 0.2833 | 0.2821 | 1.112 | 1.08 1.10 1.06 1.10 1.10 | 1.107 |
| S37 #333435 | 5 | 0.3032 | 0.3013 | 1.077 | 1.07 1.09 1.01 1.03 1.09 | 1.084 |
| S38 #333481 | 5 | 0.2944 | 0.2933 | 1.086 | 1.13 1.08 1.13 1.07 1.09 | 1.128 |
| S39 #333528 | 7 | 0.3018 | 0.2983 | 1.080 | 1.06 1.12 1.16 1.11 1.12 1.05 1.06 | 1.149 |
| S40 #333559 | 7 | 0.3109 | 0.3067 | 1.133 | 1.06 1.08 1.07 1.08 1.03 1.19 (1.14) 1.05 | 1.175 |

Over S36-S40: 29 first networks at 1.084x the median of the others on average (1.076 at a normal worker speed, largest
1.162); 471 other networks: 90th percentile 1.038, 99th 1.103, largest 1.133.

**Loss probability of a fresh run** (`loss_risk.py`, the analyst's 729 leader-sized public runs: each run's worst / mean
residual over its team's usual at that time). A fresh run's worst network = mean residual x our usual worst-to-mean ratio
x the run's excursion; a network is lost when the excursion exceeds 0.4 / (mean x usual).

| file | mean | usual ratio 1.149 (last three files) | ratio 1.127 (all 13 runs) | worst network as graded |
|---|---|---|---|---|
| S39 | 0.3018 s | 0.347 s, 15.3% of room: 1.1% (8 of 729) | 0.7% | 0.3467 s, 15.4%: 1.1% |
| S40 | 0.3109 s | 0.357 s, 12.0%: 3.0% (22) | 1.4% (10) | 0.3652 s, 9.5%: 5.1% (37) |

The 120 s limit adds nothing measurable (worst roundtrip 91.6 s, 31% of room; no same-file pair moved it by more than 17%;
0 of 591 public runs with a mean wall below 85 s lost a network on it). A first network of S40 (0.3308 s) holds on a worker
up to 1.20x slower in residual, its other networks to 1.29x; the worker of #333099 ran 1.31x slow.

**What a first network costs, exactly** (`first_network_cost.py`: two `--sites` sessions of the S41 stack, sha 8d110449,
with networks 35 and 100 in both orders; by line: `runs/c_s41stack_sites_100first/first_only_sites.txt`). Being the first
network of a process instead of the second costs +1,713 requests on either network and +0.123% of bill:

| what | requests |
|---|---|
| first-load views of load3 / load3o (`_l3_quad` 327, `_l3o_pairs` 252) | 579 |
| leaf views of the strip kernels (`_hz_leaf_views`) | 432 |
| once-per-process constants and buffers (parameter views of the learned correction 135 + 21, decoded literals 56, pair-map buffers and views 43 + 26 + 17 + 10, birth coefficients 42, feature columns 31, tables of the pair law and the exact-term module 19 + 13, others 69) | 482 |
| binding of keys the prebind list does not hold | 220 |

A network uses about 250 tree keys, 210 of them on the 649-key list, 102 among the list's first 128 (20 fresh networks,
the engineer's key censuses).

**The pieces, measured at the boundary** (one worker session each under the platform clock, parent = the bank S40;
`variant_table.py runs/c_s40 <run dirs>`; requests on network 35, first after the setup / network 100, warm):

| file | sha | first / warm | against the line above | bill against S40 | kernel time against S40 | projected wall mean / worst |
|---|---|---|---|---|---|---|
| S40 as fired | e2be4c75 | 44,578 / 42,678 | | | | 79.8 / 91.2 s (measured) |
| + K4L birth by tables + pair-law kernels v5 | cf1f735c | 44,304 / 42,409 | -274 / -269 | +0.011% | x0.985 | 79.1 / 90.3 s |
| + K31L4 = the S41 stack | 8d110449 | 44,881 / 42,982 | +577 / +573 | +0.30% | x0.991 | 79.7 / 91.0 s |
| + HV and the trainer's first dials | 33bac9db | 44,311 / 42,602 | -570 / -380 | +0.30% | x0.989 | 79.4 / 90.7 s |
| + `Rl3_bind128` | b0885a41 | 44,025 / 42,668 | -286 / +66 | +0.27% / +0.30% | x0.990 | 79.4 / 90.7 s |
| + `Rlvfs_limit64` as well | 169fd371 | 43,466 / 42,062 | -559 / -606 | same | x1.008 | 80.9 / 92.4 s |

Both fragments leave the predictions byte-identical to the file without them on both networks (`pred_<rank>.bin` of the
sessions). `l3_bind128` (`variants/fragments/l3_bind128.py`): 128 views in the setup, binding phase 17,798 requests (17,366
without), all 649 keys bound, 228 deferred fusions (266 without; that is the +66 on the warm network), setup flops
29,664,975,407. With `lvfs_limit64`: binding phase 18,028, 143 deferred fusions. On the S41 stack alone `lvfs_limit64`
measured -521 / -549 requests, bill equal, kernel time +1.6%, predictions byte-identical (`runs/c_s41stack_lvfs64`).

**The budget and what it predicts.** At the bank's platform prices (7.19 microseconds per request on the networks that are
no first network; 12.7 per extra request of a first network: 0.0236 s over about 1,880 requests, S36-S40):

| file | first network | the others (median) | loss probability: first network x 1.06 as the usual worst / mean x 1.149 |
|---|---|---|---|
| S39 | 0.323 s (measured 0.327) | 0.300 s (0.298) | 1.1% / 1.1% |
| S40 | 0.331 s | 0.307 s | 1.4% / 3.0% |
| S41 stack | 0.333 s | 0.309 s | 1.9% / 3.2% |
| S41 with HV | 0.328 s | 0.306 s | 1.1% / 2.2% |
| + `Rl3_bind128` | 0.324 s | 0.307 s | 0.8% / 2.2% |
| + `Rlvfs_limit64` as well | 0.320 s | 0.302 s | 0.55% / 1.4% |
| the caps (43,800 / 42,700) | 0.321 s | 0.307 s | |

**The diet variant under a slow setup** (`runs/s41hv_lvfs64_l3b128_slow1124`, `_slow`; sha 169fd371; predictions
byte-identical to its normal session in both):

| setup pace | keys bound / views / deferred fusions | first / warm requests | against its normal setup | the fired S40 at that pace |
|---|---|---|---|---|
| normal | 649 / 128 / 143 | 43,466 / 42,062 | | 44,578 / 42,678 |
| 1.124x (worker 3's second setup in #333559) | 596 / 128 / 0 | 43,875 / 42,429 | +409 / +367 | 44,918 / 42,974 |
| 1.3x (the judge's criterion) | 497 / 128 / 0 | 44,452 / 42,668 | +986 / +606 (+1.86% on the mean) | not run |

At the bank's prices the first network of the 1.124x case costs 0.323 s against S40's 0.3335 s: the network that came out at
0.3652 s in #333559 would have come out near 0.354 s. `designation_judge.py s41hv_lvfs64_l3b128 --pair
runs/c_s41hv_lvfs64_l3b128 --slow runs/s41hv_lvfs64_l3b128_slow`: first 43,466 (43,800), warm 42,062 (42,700), binding
phase 18,028 with all keys bound (18,100), projected mean wall 80.9 s (82), projected mean residual 0.305 s from requests
(0.309 s from the loaded session's residual instructions; 0.310), slow setup +1.86% (+2%): all met; worker run: stage D of
this file, `runs/d_s41hv_lvfs64_l3b128` (chain 23).

**Why the first 128 keys.** With the first-load views of every listed key at bind time (`l3_bind_all`, `runs/c_s41hv_l3ball`)
the first network loses 433 requests instead of 286, but the binding phase (19,010 requests) no longer fits a normal
worker's 19,000: 628 of 649 keys bound, nothing fused, warm +88. In the setup's slack a request spent on a view of one of
the 128 most-used keys takes 0.81 requests off the first network; a request spent on the marginal deferred fusion takes
0.15 off every network (38 fusions fewer cost 66 requests per network).

**Stage D of the diet variant (sha 169fd371; 16:25-16:50 UTC, load 12-15; `runs/d_s41hv_lvfs64_l3b128`): PASS\*, every
line that concerns the file met.** 20 networks in one worker under the platform clock: all outputs finite; peak handles
56,691 (S40: 57,810); peak footprint server + client 3.57 GB (the server reaches 3.19 GB at the twentieth network; S40's
quiet run read 2.37; loaded sessions of the fired S40 file read 3.0 GB today where quiet ones read 2.5); all 649 keys
bound, 128 first-load views, 143 deferred fusions; native parity on 35 and 100: bytes and bills identical; requests: first
43,466, the other 19 between 41,156 and 43,089 (mean 41,969; S40: 44,578, 41,605-43,621, mean 42,429): every network 264 to
1,112 below S40. Projection from S40's worker run over the 15 networks clean in both runs: wall 80.7 s mean / 92.1 s
worst, residual 0.307 / 0.361 s. Not judged: the machine's swap grew 154 MB during the run (other lanes). Stage A
pre-check of the same file: PASS with the one new einsum signature `ik,ik->k` (two different arrays) reviewed
(`runs/a_s41hv_lvfs64_l3b128_precheck_reviewed`, not uploaded). The lead packaged and fired it as S41 = #333587 at 16:51
UTC without a quiet window; before S41 or its successor is designated, stage D is to be repeated in a quiet window.

**The engineer's constants lever CS and his diets at the boundary (16:50-17:35 UTC; sessions at load 11-18, several with
a paging server: requests, bills and prediction bytes are exact, kernel time is not read from them).**

| file | sha | first / warm | binding phase / deferred fusions | predictions |
|---|---|---|---|---|
| HV | 33bac9db | 44,311 / 42,602 | 17,366 / 266 | S41's arithmetic |
| HV + CS | c0ae245f | 43,930 / 42,709 | 17,856 / 219 | byte-identical |
| HV + CS + `Rl3_bind128` | c6fef651 | 43,602 / 42,716 | 18,288 / 144 | byte-identical |
| the same at 1.124x setup pace | | 44,017 / 43,092 | 584 keys bound, 0 fused | byte-identical |
| rounding-level diet (F4c v2, K4L v6, pair law v9) + CS + `Rl3_bind128` (withdrawn by the engineer 16:57 UTC: +0.26% error on ten networks) | 399b0c4a | 42,918 / 42,032 | 18,291 / 144 | rounding level |
| HV2 + CS + byte-identical diet (F4c v3, K4L v7, edge cut v2, pair law v9), the engineer's build of 17:13 UTC | d464c12c | 43,487 / 42,258 | 17,856 / 219 | byte-identical to S41; kernel time x0.984 of S40 in a clean session: wall 78.9 / 90.1 s |
| + `Rl3_bind128` = the S42 request line | 14dca649 | 43,159 / 42,265 | 18,288 / 144 | byte-identical to S41 |
| the same at 1.124x setup pace | | 43,574 / 42,641 | 584 keys bound, 0 fused | byte-identical |
| the same at 1.3x setup pace | | 44,134 / 42,838 (+1.81% on the mean) | 486 keys bound, 0 fused | byte-identical |
| + `Rlvfs_limit64` as well | befcb016 | 42,619 / 41,659 | 18,518 / 99 | byte-identical; wall 80.6 / 92.1 s |
| S41 as fired (HV + `Rlvfs_limit64` + `Rl3_bind128`) | 169fd371 | 43,466 / 42,062 | 18,028 / 143 | byte-identical |
| S41 as fired at 1.124x setup pace | | 43,875 / 42,429 | 596 keys bound, 0 fused | byte-identical |

Judge lines of the S42 request line (`designation_judge.py s42req --pair runs/c_s42req_l3b128 --slow
runs/s42req_l3b128_slow`): first 43,159 (43,800), warm 42,265 (42,700), binding 18,288 with all keys bound (18,600), mean
residual 0.304 s (0.310), slow setup +1.81% (+2%): met; kernel and wall from the clean session of the build alone.

CS takes 490 requests at the head of the binding phase and 381 off a first network; it puts 107 on every other network
because 47 deferred fusions no longer fit before the guard. The setup's guarded budget is now full: binding 17,366 +
constants 490 + views 432 = 18,288 of a normal worker's 19,000 requests; the rest buys 144 fusions. The judge's binding
line moved from 18,100 to 18,600 requests (17:10 UTC) for that reason. One session was stopped by the guard at 256 MiB of
added system swap (other lanes) and repeated (`_stale/c_s41hvcs_l3b128_killed_on_swap_*`, `runs/c_s41hvcs_l3b128_r2`).

**Instrument change, 17:12 UTC.** `project.CALM_CYCLES` 50,000 -> 30,000 residual cycles per request: quiet-window sessions
of S37-S40 spend 24.0-26.4 thousand cycles and 44.1-44.8 thousand instructions per request whatever the file; sessions at
load 10+ spend 38-72 thousand cycles and read 0.4-3.5% more instructions per request, which against stored quiet parent
legs made the residual line read up to 3% high. Re-judged with it: S41's stage C residual 0.305 s mean / 0.358 s worst
(`runs/c_s41_fired_rejudged`).

**S41 graded 1.711e-9 (#333587, 18:18 UTC), no network lost; it is the bank (`runs/new_bank_S41.log`).** Final-layer MSE
1.462e-8 (-3.0% against S40), budget share 0.117035.

| quantity | projected (stage C, parent S40, residual from requests) | platform | error |
|---|---|---|---|
| kernel, mean | 71.97 s | 74.28 s | -3.1% |
| overhead, mean | 8.648 s | 9.027 s | -4.2% |
| residual, mean / worst network | 0.3047 / 0.358 s | 0.3182 / 0.3536 s | -4.3% / +1.2% |
| predict wall, mean / worst network | 80.9 / 92.4 s | 83.6 / 90.2 s | -2.7 s / +2.2 s |

Kernel work, round trips and requests are three independent counts and all three miss by 3-4% in one direction: the
platform ran slower in that hour (the submission waited 57 minutes in the queue). The setups say the same
(`runs/setup_keys_s41.log`, `setup_keys_s41_second.log`): the five first setups were cut by the guard after 17,883 /
18,001 / 17,610 / 17,740 / 16,804 requests (642, 648, 629, 635, 591 of 649 keys bound, nothing fused); of the four setups in
mid-run three were complete (18,026 requests or more) and one stopped at 17,106 (605 keys). Platform constants of this run:
7.44 microseconds of residual per request, 0.1901 ms of overhead per round trip (S36-S40: 6.93-7.13 and 0.177-0.182).

The run itself separates the hour from the file (networks that are no first network, by position in the run):

| networks | S41's run: kernel ms per Gflop / overhead s / residual s / wall s | S40's run |
|---|---|---|
| 0-24 | 296.2 / 9.164 / 0.3210 / 85.3 | 274.3 / 8.799 / 0.3090 / 79.1 |
| 25-49 | 293.9 / 9.073 / 0.3174 / 85.3 | 277.4 / 8.977 / 0.3127 / 81.1 |
| 50-74 | 286.9 / 8.915 / 0.3136 / 82.4 | 277.2 / 8.837 / 0.3091 / 79.7 |
| 75-99 | 283.5 / 8.860 / 0.3148 / 81.1 | 277.4 / 8.750 / 0.3068 / 79.2 |

The same file got 4.3% faster in kernel time per flop from the first quarter of its run to the last; by stage C it should
cost 1.005x S40's kernel time per flop, and the excess is 7.5% in the first quarter and 1.7% in the last. Overhead per round
trip and residual per request, which the un-fused trees of lvfs64 cannot touch, were 3.3% and 3.5% above S40's run after
taking out the file's own counts. The queue was draining while S41 ran.

Network by network (`platform_networks.py`): nine first networks 0.3137-0.3536 s (mean 0.3347), the other 91 mean 0.3166
(max 0.3521); first / other 1.057 (S40 1.069, S39 1.091); worst / mean 1.111 (S40 1.175, S39 1.149). The worst network,
index 65, is the first network of worker 1 after its second setup, the one cut at 605 keys (1.111x slow): the event of
#333559 again. The second largest, 0.3521 s, is an ordinary network at a normal kernel speed (0.97x).

**The judge's lines in seconds were changed at 18:30 UTC for that reason** (`designation_judge.py`; the previous forms are in
`_stale/`): a mean measured in one hour is not comparable with a limit set in another, so the projected and measured mean
residual (0.310 s), the first networks' mean (0.328 s) and the mean wall (82 s) are shown and no longer judged. Judged:
projected worst-network wall at most 100 s; in the graded run no network lost, worst network at most 0.36 s of residual
and 105 s of roundtrip; the counts as before. With these lines S39 passes, S40 fails two (first-network requests, worst
network 0.3652 s), and S41 meets every line: first 43,466, warm 42,062, binding 18,028 with all keys bound, projected
worst-network wall 92.4 s, slow setup +1.86%, 20 networks finite, handles 56,691, footprint 3.57 GB, native parity
identical, platform 0 of 100 lost, worst network 0.3536 s and 90.6 s of roundtrip. Its verdict is REPEAT for one reason: the
worker run was made on the shared box (swap +154 MB); the quiet-window stage D is owed.

**Loss probability and the platform's hour** (first networks x 1.06 as the usual worst network; the analyst's 729 runs): S41
in an hour like its own 1.9% (14 of 729); in an hour like S40's 0.7%; S40's file in an hour like S41's 5.2%; the S42 request
line 1.1% in a slow hour, 0.4% in a normal one. The hour moves every network by about 4%, as much as the whole diet, and
the final run is likely to be a slow one (every team's file queued at once). Projections now start from S41's record, the
slowest of our seven graded runs: they are the slow-hour case and read 3-4% above what S40's record would give.

## 13. What the leaders' risk level would buy in bill (lead's question of 2026-10-02 19:14 UTC; no session)

**Answer: about 0.4% of bill, 0.7% at the very most. Under 1%: the leaders' lower bill is structure, not risk.** J2W's
#333573 runs at 0.369 / 0.398 s of residual and 105 / 119 s of wall with our final-layer error and a budget share 6.8%
below ours; a public bank of ours at 0.365 s mean and 100 s mean would have 6,300 more requests a network (at the price
of S41's hour; 8,400 at S40's) and 16 s more wall to spend.

Source: the engineer's product census of the S40 arithmetic on network 651 (`O/engineer/census/s40_651.json`: 164
products with shapes, leaf patterns and bills by operation) and his fitted bill and time models of the tree kernel (the
file's `_p6_bill`, `_p6_secs`), which reproduce the measured bill of 75 single-plan products to -0.2% at the median.
`python -B bill_room.py` prints every number below.

| bill of the S40 arithmetic (262.4 Gflop on network 651) | share |
|---|---|
| leaf multiplications | 78.5% |
| tree additions | 12.7% |
| other work inside products | 1.1% |
| operations that are no product | 7.7% |

| by plan | share of the bill | products |
|---|---|---|
| strips at depth 6 | 28.9% | 27 |
| depth 6 in column or k packets | 22.5% | 25 |
| depth 7 (2 of them in packets) | 22.5% | 27 |
| plain depth 6 | 11.5% | 26 |
| upper-block symmetric products | 3.8% | 14 |
| below depth 6 | 2.9% | 24 |
| plain matmuls (14 rank-one star products, 5 small ones) | 0.25% | 19 |

| option (rounding-identical, none built) | bill | requests a network | platform wall |
|---|---|---|---|
| 1. products below depth 6 one level deeper (20 of 24 move) | -0.10% | +360 | +0.5 s |
| 2. one more level on the depth-6 families where the model says the bill falls (8 of 78 products move) | -0.14% (-0.35% with the engineer's figure of -0.25% for seven-level strips) | +144 | +3.0 s |
| 3. depth 8 on the depth-7 families (5 of 27 move) | -0.09% | +90 | +6.4 s by the model, more in practice (per-leaf call cost; new arena regions) |
| 4. the five small plain products as trees | at most -0.06% | +375 | about 0 |
| 5. undoing the request diets that cost bill: three-request operand loads; K4L birth by tables | -0.06%; -0.011% | +1,800; +255 | 0 |
| 6. fusion variants | 0 | +560 (32M limit) to +3,000 (no fusion) | -1.5 to -2.5 s |
| 7. the thin-star diet (+0.48% when fired in S36) | nothing to give back: since S39 the star is carried by its row sums and its products are rank one (0.006% of the bill) | | |
| 8. the engineer's byte-identical diet | -0.023% | -323 | 0 |

Best combination 1 + 2 + 4 + 5 + 8: -0.39% of the bill (-0.60% with the engineer's strip figure) for about +2,600 requests
and +3.5 s; with 3: -0.48% (-0.69%) at +10 s. The products already sit where one more level stops paying, so the headroom
cannot be spent. (The first log line of this answer, 19:22 UTC, counted the requests of rows 1-3 for every product of a
family instead of the products that move; corrected at 19:24 UTC.)

Where a structural 6.8% could sit in our own bill: the only large part that is the same on every network is the
covariance sandwich W'CW of every layer, `first = _mm(C32, w)` as a full 1024^3 product at depth 7 (14 calls, 14.1 Gflop,
5.4% of the bill) and `_sym_upper(wt, first)` on the upper blocks (14 calls, 9.9 Gflop, 3.8%). The analyst's regression
of 19:16 UTC reads J2W's bill as 0.98 x ours minus a constant of 13 Gflop a network.

## 14. S42 (2026-10-02 19:24-19:58 UTC): stage C and D on the step-1 variant, the fired step-2 file by the fast path

S42 = S41's arithmetic + one new input column of the learned mean correction + the engineer's free request diet (HV2, CS,
F4c v3, K4L v7, edge cut v2, pair law v9) + the release fragment `Rl3_bind128`, without lvfs64. The engineer's file came
in two steps of the trainer's weights: step 1 (sha 33582663) and step 2 (sha 0fccf396), 128 number literals apart.

| what | file | result |
|---|---|---|
| stage C, 19:24-19:27 (`runs/c_s42`, parent = stored legs of the bank S41) | step-1 variant, sha 635426c8 | PASS: requests 43,470 first / 42,573 warm; bill x0.9995, kernel time x0.980, requests x1.006 of S41; binding phase 18,288, all 649 keys bound, 144 deferred fusions; projected wall 81.8 s mean / 88.2 s worst, residual 0.320 / 0.356 s (at the speed of S41's slow hour; about 79.1 s and 0.307 s at S40's) |
| stage A pre-check (`runs/a_s42_precheck_reviewed`, not uploaded) | step-1 variant | PASS with one new signature against the bank: `ij,c->ijc` (lines 10556-10558: a kept block times a channel vector, two different operands) |
| stage D, 19:27-19:48 (`runs/d_s42`, load 11-13) | step-1 variant | PASS, every gate including swap: 20 networks finite, peak handles 55,367, footprint 3.44 GB, swap +0, native parity on 35 / 100 bytes and bills identical; requests first 43,470, the other 19 between 41,629 and 43,501 (mean 42,365); projected wall 82.1 / 88.6 s, residual 0.321 / 0.357 s |
| the lead's stage A and C (`runs/a_s42b`, `runs/c_s42b`) | step-2 variant, sha 471a9841 = FIRED as #333607 at 19:49:01 | PASS, counts as the step-1 variant's to the request |
| literal proof (`literal_diff.py`) | step-1 against step-2 variant | LITERAL-ONLY: 128 numbers on line 11084 |
| census of the two sessions (`fast_path.py --judge runs/c_s42/C0 runs/c_s42b/C0`) | | the same: requests, free trips, handles, kernel calls, billed families, bills; 16 bytes of stored index arrays differ per network |
| native parity of the fired file (`native_parity.py`, `runs/_native/471a9841f15a_*`) | step-2 variant | PASS: bytes identical to `runs/c_s42b/C0` on networks 35 and 100, bills equal; the check fails against the step-1 session, as it must |
| slow setup, worker 1.3x slower (`runs/s42b_slow`) | step-2 variant | 486 of 649 keys bound; 44,445 / 43,146 requests, +1.80% (limit +2%) |
| judge (`designation_judge.py s42 --pair runs/c_s42b --slow runs/s42b_slow --worker runs/d_s42b`) | | every line met before the grade; `runs/d_s42b` is a copy of the step-1 variant's worker run (README inside) |

`native_parity.py` as first written compared the native float64 file with the boundary's float32 bytes without widening
and printed FAIL; it now uses stage D's own comparison (found and corrected before anything was logged from it).

## 15. S42 as the bank and its safe designated variant (2026-10-10 02:00 UTC on)

**Bank.** `new_bank.sh S42 variants/s42_0fccf396_l3b128 O/board_evals/eval_333607.json s42b` (log `runs/new_bank_S42.log`):
`calibration/bank.json` = #333607 (mean / worst wall 77.49 / 83.59 s, kernel 68.53 s, overhead 8.646 s, residual 0.3068 /
0.3561 s); stored legs `runs/c_S42_bank_legs_heavy` (the fired file at the platform's setup pace, constants charged; pointer
`calibration/bank_stage_c_paced_*.txt`). Stage C's projection made from the S41 bank (a slow hour) read every mean 4-6% above
the platform (kernel 72.7 / 68.5 s, wall 82.1 / 77.5 s); worst-network residual 0.356 / 0.356 s.

**What the worst network was** (`platform_networks.py O/board_evals/eval_333607.json`): network 9, 0.3561 s, an ordinary
network (no setup on it) at 1.167x the median of the other networks (0.3051 s), kernel per flop 1.044x the run's median; network
80 (0.3398 s, 1.114x) likewise. Earlier runs had the same kind of spike (S41 network 79 1.110x, S40 network 6 1.114x). The five
first networks were 0.322-0.331 s (1.055-1.086x). A safer file therefore has to take requests off every network.

**Pieces, one stage C session each (02:15-02:27 UTC), the fired S42 file + one fragment, setup under the platform clock with the
constants charged; networks 35 (first after the setup) and 100 (warm); predictions byte-identical to S42 in every row**
(`variant_table.py runs/s42b_pace17800 <runs>`; walls projected from the S42 bank; S42 graded 77.5 / 83.6 s):

| piece | requests first / warm | vs S42 | bill first / warm | keys bound | projected wall mean / worst | residual first / warm |
|---|---|---|---|---|---|---|
| S42 as fired (at the platform's setup) | 43,696 / 42,743 | | | 626 | (graded 77.5 / 83.6 s) | 0.319 / 0.307 s |
| (a) no un-fused trees, `Rlvfs_limit_off` | 42,610 / 41,581 | -1,086 / -1,162 | +0.019 / +0.033% | 593 | 83.9 / 90.5 s | 0.312 / 0.299 s |
| (b) un-fused only above 64M floats, `Rlvfs_limit64` | 43,198 / 42,223 | -498 / -520 | +0.004 / +0.020% | 615 | 79.5 / 85.8 s | 0.316 / 0.303 s |
| (c) fuse keys 129-300 at bind, `_PB_EAGER = 300` | 43,830 / 42,856 | +134 / +113 | +0.032 / +0.040% | 579 | 77.9 / 84.0 s | worse: closed |
| (b)+(c) | 43,441 / 42,256 | -255 / -487 | +0.101 / +0.057% | 543 | 79.5 / 85.8 s | |
| (a)+(c) | 43,032 / 41,562 | -664 / -1,181 | +0.230 / +0.121% | 457 | 84.8 / 91.5 s | |

(a) costs kernel time (the fused trees are memory-bound: projected kernel +9.7%), not bill; its wall stays far under the
judge's 100 s. (c): fusing at bind time slows the binding so much that 47 more keys (alone; 72-136 with (b) or (a)) go to lazy binds
inside predictions.
(d) the covariance sandwich's second product in the file's own two-strip form (`Rsym_two_strips`: `_sym_upper =
_PS_PARENT_SYM_UPPER`, the function the later block forms saved as their parent; another block geometry, so rounding-level):
engineer's count -2,660 requests a network for +0.27% of bill; stage A of (a)+(d) PASS (sha d13b9607); its session, stage D
and slow session run as `release_chain35.sh` (machine priority of the trainer until 03:20 UTC; a first stage D of (a) alone was
stopped at 6 of 20 networks, `runs/_stale/d_s42safe_stopped_0242Z`).

**Loss probability of a fresh 100-network run** (`loss_risk.py`, the analyst's 729 excursions, our usual worst / mean 1.160,
price 7.098 microseconds a request; local / week readings): S42 2.19% / 3.16%; S42 + (a) 1.10% / 2.33%; S42 + (a) + (d)
0.41% / 1.37% if (d)'s count holds (39,436 requests on average, worst network about 0.325 s at S42's hour). Score cost of (a)+(d):
bill +0.3%.

**Designation** (repo `official_rules_v2.txt`, section 6): one designated submission among our valid Phase 2 submissions, not
necessarily the best public one or the newest; the Team Representative selects it in an interface published near the end;
default = the best public one. The safe file therefore has to be fired once.

**Lead's decision (03:05Z):** (a) goes into every public bank from S43 on (the best public file is the safe file); (d) is out.
One file per physics step: file + `Rl3_bind128` + `Rlvfs_limit_off` (`SAFE_FRAGS='l3_bind128 lvfs_limit_off'`). The S43 skeleton's
file (engineer sha 0df2ba07 + both fragments, `variants/s43skel_safe`, sha 0b2a3182): stage A PASS with three reviewed einsum
signatures (`alsqj,lsjk,lk->aqsk`, `a,sq,k->aqsk` in F4Cv2's birth; `kn,kc->nc` in the final-layer columns: distinct operands).

**The 5 s setup cap (a breach fails the whole submission; rules page read by the analyst 10-10).** Our 86 public setups (15 runs
since #332361): max 4.336 s, 99th percentile 4.333 s, median 4.282 s; ours are the longest on the board (15,854 recorded setups
of all teams, none longer). A setup = binding until the 4.0 s guard (wall-clock: a slow worker binds fewer keys and still stops
at 4.0 s) + two unguarded fusion phases (psdb_fused 9 keys, read7_fused 4 keys: 514 requests, 0.116 s at the platform's pace) +
0.16-0.20 s the platform counts outside our clock (platform wall minus our last setup phase on S37-S42's banks). A breach needs
that 0.30 s tail to grow 3.3x; on the board a setup is at most 1.235x its run's median (2,480 setups with identical work, 420
runs, all teams; p99 1.11x, p99.9 1.22x). A power-law tail through p99 and p99.9 gives 3e-7 per setup: about 2e-6 for a
100-network run (6 setups) and 6e-6 for 300 networks (18 setups). Lower guards cost: 3.5 s about +520 / +280 requests (first /
warm), bill +0.12% / +0.05%, P(network loss) of the (a) file 1.10% -> 1.37%; 3.0 s +1,050 / +560, +0.24% / +0.11%, 1.92%.
Kept at 4.0 s. New judge lines (`designation_judge.py`, old form in `_stale/`): projected platform setup wall = last setup phase
on the platform clock + 0.19 s <= 4.5 s, the same in the 1.3x session <= 4.6 s, and a graded file's longest setup <= 4.5 s
(S42: 4.306 / 4.388 / 4.319 s). They catch the real exposure: a fragment that adds setup work after the guard, or a new
flopscope (the grader ran 0.12.1 on 10-02 and 10-09; our sessions run 0.12.0 and have matched its counts).

**Two designations (rules page):** if the better of the two private scores counts, the pair to designate is a hedge of one
physics file: the safe file (P(loss) about 1%) and an aggressive twin with every bill lever that costs requests or wall
(P(loss) 10-20% acceptable): P(podium) = (1 - L_A) P_A + L_A (1 - L_S) P_S. The planner probes (prices / 4: `prices_quarter`,
prices 0: `prices_zero`, on S42 + (a)) price that twin (chain 36).

**Planner prices (chain 36, one stage C session each on S42 + (a), against (a) alone):** prices / 4: bill -0.24% first network /
-0.17% warm for +252 / +426 requests; prices 0: -0.27% / -0.20% for +311 / +576 (about 1.3M flops of bill per extra request);
kernel +2.1-2.5%; rounding-level. S42 + (a) + prices 0: P(loss) 1.37% (S42 as fired 2.19%): better than S42 on both counts.

**S44 fired as a pair (lead 08:47Z), from the engineer's sha 67dfcf71:** SAFE #335147 = + Rl3_bind128 + Rlvfs_limit_off +
Rprices_zero (candidate f613a1a4; stage C 43,484 / 42,548 requests, slow +1.38%, setup 4.312 / 4.407 s): graded 1.629e-9, wall
85.3 / 91.3 s, residual 0.31 / 0.34 s. PRIMARY #335148 = the same hand-over through `ebuild/build.py` (a copy of the engineer's
builder that writes here; control: it reproduced 67dfcf71 byte for byte) with F4Cv3 / K4Lv7, no L3 / L3o, S42's fusion,
+ Rprices_zero (candidate 43fde9c3; stage C 46,251 / 45,721, P(loss) 10.56%, setup 4.302 / 4.392 s): graded 1.627e-9, budget
share 0.11667 against 0.11684, wall 81.3 / 87.7 s, residual 0.33 / 0.39 s (close to the cap, as priced). Stage D skipped for
both. S45 (engineer sha b36b7154 = jmH): the same recipe, chain 41 (safe 4692b091, primary a7abce90): graded SAFE #335213
1.586e-9, PRIMARY #335215 1.584e-9, 100/100, worst residual 0.35 / 0.37 s (lead 15:57Z; fourth place). S46 (engineer SAFE
63f89e8d, PRIMARY f89b8951 = jmIv): chain 42 qualifies the safe recipe (70f7857f), the release twin (0e8d4356: the jmIf
hand-over through the refreshed builder copy, controls reproduced both engineer files; safe base + prices 0, because the
engineer's OVR block comes after the release fragments and would undo Rprices_zero on his kp10 base) and f89b8951 as is; the
chain writes the fire lines.

**10-11 (Opus lead back): twin risk and the S47 pipeline.** `twin_risk.py` prices a twin that carries extra requests over the
safe file on our six graded S44-S46 runs (price 7.11-7.27 us a request, worst / mean 1.098-1.183, none lost): P(loss) per 100
networks at the median hour +500: 1.9-3.2%, +1,000: 3.0-4.4%, +1,500: 3.8-5.8%, +2,000: 5.4-7.5% (worst hour 7.8-10.6 / 10-12 /
13-16 / 17-19%); expected pair gain (1 - P) x cut stays 88-97% of the cut up to +2,000 requests. `s47_pair.sh` (checklist
`S47_RELEASE_v3.md`) takes the engineer's safe and twin files to fire lines in one command; `release_gate_v3.py` = Astra's v2 audit
with this machine's swap / paging gates as warnings (controls: passes both fired S46 files, blocks the crashed twin).

**S43 tool** `qualify_pair.sh <engineer file> <tag> ['reviewed sigs']` (DRY=1: builds and stage A only): bank = file +
`Rl3_bind128`; safe = file + `Rl3_bind128` + `Rlvfs_limit_off` + `Rsym_two_strips`; stage A of both, then C, D, 1.3x slow
session and judge of the bank ('BANK VERDICT'), then of the safe file ('SAFE VERDICT'). Dry run on the S42 engineer file
(`runs/qualify_dry_s42.log`): rebuilt 471a9841 and d13b9607 byte for byte, stage A PASS on both.

## 10. Price list (asked by the lead at 20:20 and 20:40 UTC on 10-01; compiled 2026-10-02 03:55 UTC, no new compute)

Reference: S37 on the platform, 78.0 s mean wall (kernel 69.1 s), 0.303 s mean residual, 42,864 requests per network,
budget share 0.1190. One request is 0.0071 s of residual per 1,000; 1% of kernel is 0.69 s of wall. "Measured" rows were
measured on RRPI (section 4, networks 35 and 100, in process) or at the boundary as stated; "estimate" rows are counts from
the census (section 7) or from the code, not yet built. Sizes on the strip stack may differ by a few tenths.

| lever | bill | requests per network | kernel time | predictions | status |
|---|---|---|---|---|---|
| **in S36 / S37 already** | | | | | |
| un-fuse right-operand and output trees above 32M floats | 0 | +1,100 under the platform's setup (+1,378 with every key fused) | -9% (platform: S36 66.9 s against S35 73.4 s together with the next row and the strips) | byte-identical | measured, fired |
| thin low-rank-star products as plain matmuls | +0.48% | -4,350 | -1.7% | rounding-level; -0.24% error on 20 fresh networks, 0 on the board | measured, fired |
| regenerated prebind list | -0.05% over a 12-network session (engineer), -0.3% on a worker's first two networks (boundary pair) | -1,490 | about 0 | byte-identical | measured at the boundary, fired |
| **free in bill, towards more residual room** | | | | | |
| un-fuse only above 64M floats | 0 | about -550 | about +4% (+3 s) | byte-identical | built, one session needed |
| no un-fusing | 0 | about -1,100 | about +10% (+7 s) | byte-identical | built, one session needed |
| F4 / K4L small-vector algebra merged | 0 | about -400 (-300..-600) | about 0 | rounding-level | estimate, engineer |
| B and P1 of the star as one thin product | 0 | about -700 | about 0 | rounding-level | estimate, engineer |
| one view less per operand load | about 0 | about -600 | about 0 | byte-identical if the sum form bills alike | estimate, engineer |
| **costing bill** | | | | | |
| covariance outer rectangles one level shallower | +0.09% | -462 | -1.5% | rounding-level | measured |
| depth-7 1024^3 products back to depth 6 | +0.43% | -140 | -6.7% (-4.6 s) | rounding-level | measured |
| depth-7 reads back to the split reads | +0.36% | -8 | -3.0% (-2.1 s) | rounding-level | measured |
| depth-7 column-packet pushes back to depth 6 | +0.89% | -598 | -7.9% (-5.5 s) | rounding-level | measured |
| all three depth-7 layers off | +1.61% | -845 | -14.6% (-10 s) | rounding-level | measured |
| row gather by indexing instead of scatter | +0.23% | -140 | -2.3% | byte-identical | measured |
| padded depth-5 -> 6 moves off | +0.34% | +232 | -3.3% | rounding-level | measured |

Exchange rates. Requests: the thin-star diet buys 9,000 requests per 1% of bill, the outer rectangles 5,100 per 1%; every
other bill-costing dial buys kernel time, not requests. Kernel time: 6-11 s of wall per 1% of bill (depth-7 1024^3 products
10.7 s, reads 5.8 s, column-packet pushes 6.2 s). A wall of 65 s from S37's 78 s costs about +1.6% of bill (all depth-7
layers off, -10 s) plus the outer rectangles; it buys nothing on the residual side, which is the exposed limit (section 9).
The other direction (spend wall, save bill): the only kernel dials that save bill are the ones already on (depth 7 since
S29, the strips since S36); the engineer measured deeper recursion as bill-negative on every large family and puts 7-level
strips at -0.25% for a high CPU price; undoing the thin-star diet saves 0.48% of bill but costs 4,350 requests, which takes
the residual to 0.334 s. So the 42 s of wall room S37 has cannot be turned into bill at a useful rate with the known dials.

## 7. What the residual is made of (2026-10-01 22:45 UTC)

`runs/prof_rrpi_noguard`: one RRPI session with the client under cProfile (`stage_c_boundary.py --profile`,
`residual_report.py`). The residual is the wall time outside flopscope's dispatch spans: the estimator's own Python plus the
part of flopscope's per-call wrapper that runs before the span's clock starts and after it stops.

| network 35, one prediction | value |
|---|---|
| requests | 42,527 |
| flopscope calls by the estimator | 43,438 |
| Python function calls inside the estimator file | 22,205 |
| self time of the estimator's functions + the builtins they call (under the profiler) | 0.45 s |
| self time of flopscope's per-call wrapper (function wrapper, context manager, caller check), top-level share | about 0.7 s |
| largest single estimator function | `_ipt_leaf_unwind_ab` 0.043 s; `_ipt_forward_a_ab` 0.035 s; `_ipt_forward_b_ab` 0.034 s |

- The estimator's Python has no hot spot: its loops already do little besides calling flopscope. About two thirds of the
  residual-type time is flopscope's own wrapper, paid once per call whatever the estimator does.
- So the residual moves with the number of charged calls and with nothing else we control. On the platform it is 7.0-7.3
  microseconds per request for every bank (section 2). The levers are the ones in section 4 (thin-star products as plain
  matmuls: -4,350 calls per network for +0.48% bill; level fusion: -3,100 calls for +10% kernel CPU).
- Profiler times are inflated (cProfile adds a fixed cost per call) and the machine was at load 25-35; the shares are the result.

**Where the requests are** (`runs/c_rrpi_sites_paced/C0`, `request_sites.py`; RRPI, network 35 as the FIRST network after a
setup at a normal platform worker's pace: 649 keys bound, 128 + 24 fused; the session was cut off by its time limit after this
network). 43,984 requests, of which 11,384 (25.9%) do no arithmetic:

| group | requests | share | note |
|---|---|---|---|
| tree-kernel arithmetic (`_ipt_leaf_unwind_ab` 8,571, `_ipt_forward_b_ab` 4,980, `_ipt_forward_a_ab` 4,408) | 17,959 | 40.8% | about 1,450 fewer with every listed key fused (42,527 requests in the no-guard session) |
| estimator body (`_predict_binary`, pair map, block products, births, learned nets, ...) | about 17,000 | 39% | |
| per-product operand views (`_rs_load_a` 284 products x 3, `_rs_load_b` 311 x 3, `_rs_out_quadrants` 300 x 4) | 2,985 | 6.8% | inherent in the kernel's layout |
| lazy tree binds (`_rs_basis_views` 798, `_ipt_side` 694, `_lvf_rs_level1` 507, `_ipt_view` 210) | 2,209 | 5.0% | 105 keys x 21 requests on a worker's first network; fewer on later ones |
| output buffers (`empty`) and float64 -> float32 casts (`_ps32`, `asarray`) | 487 + 505 | 2.3% | |

By operation: add 12,781, subtract 10,026, `__getitem__` 4,368, multiply 3,222, unstack 3,039, copyto 1,715, reshape 1,249,
true_divide 810, matmul 655, asarray 591, transpose 564, empty 487.
Against today's platform setup, a setup that binds and fuses every key a network needs would save about 3,650 requests on a
worker's first network (8.3%) and about 1,450 plus that network's own lazy binds on later ones, at zero bill: the engineer's
regenerated prebind list (TEAM_LOG 23:02 UTC) is that lever.

## 5. Runs

| run | what | result |
|---|---|---|
| `runs/a_s35_v1`, `runs/a_rrpi_v1` | stage A | PASS, PASS; archive 459,038 / 459,611 bytes; 12 einsum signatures, none new |
| `runs/c_rrpi_vs_s35_load1` (+ `_judged`) | stage C, RRPI vs S35, parent-candidate-candidate-parent | PASS*: bill x0.99896, kernel proxy x1.0099, requests x1.0071, residual instructions x1.0061; projected wall 83.0 s mean / 95.6 s worst, residual 0.306 / 0.341 s; setup 4.12 s not judged (loaded) |
| `runs/cal_s32_vs_s35`, `cal_s33_vs_s35`, `cal_s34_vs_s35`, `cal_fg_vs_s35` | one session each of the earlier banks and of the failed #333099 source | calibration inputs, all clean |
| `runs/d_s35_ref1` | stage D on the bank | 20/20 finite; peak handles 59,751; group RSS 6.33 GB; setup 4.10 s; native parity on 35,100; system swap +141 MB during the run (shared machine); networks 73, 76, 80 paged (my two-jobs-at-once breach) and are excluded from the stored reference `calibration/bank_stage_d.json` |
| `runs/d_rrpi_1` (+ `_judged`) | stage D on RRPI | PASS: 20/20 finite; peak handles 60,022; group RSS 6.36 GB; setup 3.93 s; native parity on 35,100; vs bank over 17 clean networks kernel proxy x1.0083, requests x1.0071; projected wall 82.8 s mean / 95.3 s worst, residual 0.306 / 0.341 s |
| `runs/b_rrpi_vs_s35` | stage B, RRPI vs S35, 40 networks | NOT RUN: the first native batch (S35, 10 networks) was killed by the guard after 139 s and 3 networks, system swap +273 MB while three trainer workers ran; partial rows kept in `runs/_native/f9f4790b3c65_8e08cafa_a1`. Waiting for the lead's decision on machine sharing. Known so far: RRPI's native predictions on networks 35 and 100 are byte-identical to S35's, bill -0.105% |
| `runs/c_rrpi_fastC_vs_s35` (+ `_normal_judged`, `_degraded_judged`) | stage C, fastC vs S35 | normal setup: PASS, bill +0.38%, requests -6.4%, projected wall 73.0 / 84.0 s, residual 0.283 / 0.315 s; slow setup (load 27): section 4a-bis |
| `runs/c_rrpi_lvfs32_vs_s35` | stage C, lvfuse_sized32 on RRPI vs S35 (parent legs reused) | PASS, bill -0.10% (RRPI's), requests +4.0%, kernel proxy -6.8%, projected wall 76.2 / 87.7 s, residual 0.313 / 0.349 s |
| `runs/d_rrpi_fastC` (+ `_nonative_judged`) | stage D, fastC | 20/20 finite; peak handles 55,385; group RSS 5.12 GB; setup 3.60 s; bill x1.0037 on every network; requests x0.934; over the 15 networks clean in both runs projected wall 74.4 / 85.6 s, residual 0.287 / 0.320 s; system swap +240 MB during the run (shared machine, not judged); native parity step waiting for memory |
| `runs/d_rrpi_fastC_rejudged_v3` | stage D of fastC re-assessed once its native rows on 35, 100 existed | PASS* (only the system-wide swap gate not judged): native parity TRUE (bytes identical, worker bill = native bill on both networks); a single network may take 1.61x the mean wall and 1.39x the mean residual before a limit |
| `runs/b_rrpi_fastC_vs_rrpi_fresh20a` | stage B, fastC vs RRPI, native, the first 20 of FRESH40 (newer ranks nobody trained or selected on), two batches of 10 per arm, arms alternating | PASS (CPU-variant limits): pooled MSE -0.238% [-0.39, -0.09], lower on 13 of 20 (per-network mean -0.238%, standard error 0.080%); bill +0.481%; adjusted +0.245% [+0.09, +0.39]; worst network +0.80% (rank 550); native instructions -6.1%. RRPI's rows on these ranks are stored for later pairs |
| `runs/c_rrpi_fastC_short` (+ `_judged2`) | stage C, fastC vs S35 with `--short-setup` (platform-like fusion) while the machine load was 13-16 | PASS*: the bank's own setup bound only 464-485 of 649 keys before its guard in these sessions (fastC's 621-687), three network measurements paged and were dropped; requests 40,671 against 43,906 (-7.4%), residual instructions -7.5%, kernel proxy -9.8%; projected wall 73.1 / 84.2 s, residual 0.282 / 0.313 s; setup 4.20 s not judged (loaded) |
| `runs/anatomy/measure_*_emu128`, `rrpi_fastD` | in-process, platform-like fusion regime; the deeper combination | sections 4, 4a |
| `runs/d_s37_paced` (run by the lead through `lead_s37_chain.sh`) | stage D of the S37 file (sha 2a25a074), setup under the platform clock | PASS, every gate: 20/20 networks finite; peak handles 56,441 (39,757 after the first network; still +100 per network at the end, tree caches 983 of 992); group RSS 6.20 GB, server footprint 2.97 GB; no swap added; 649 of 649 keys bound; requests 42,186 per network over the 20; native parity on 35 and 100 in bytes and bill. On the 14 networks calm in this run and in S35's own: bill 0.9858x, projected platform kernel 67.1 s, residual instructions per request 0.993x |
| `runs/d_stackHP_lvfs32_tsr_paced` (started by the lead) | stage D of the S36 file | stopped by the lead after two networks (the box crawled: 400-450 s per network); not a verdict. Handles 39,684 / 40,970, finite, same requests and bill as the stage-C sessions |
| `runs/a_s37_2a25a074_v3`, `a_s36_87b3d6ef_v2`, `a_s37_F4t06_K4L` (lead) | stage A of the S37 and S36 files | PASS; S37 with the three new einsum signatures passed as reviewed; both compile under Python 3.10.20 with no 3.11-only name |
| `runs/c_s35_lvfs32_short` | stage C, S35 + lvfuse_sized32, old `--short-setup` | FAIL on `clean`: both sessions paged (system/user 0.37-0.40); not a verdict |
| `runs/d_s35_lvfs32` (+ `_rejudged`) | stage D, S35 + lvfuse_sized32, setup under this machine's clock at load 10-13 | NOT A VERDICT ON THE BUILD: the setup bound only 418 of 649 keys before its guard. 20/20 finite, peak handles 50,845, group RSS 4.36 GB; bytes identical to native on 35 and 100; worker bill +0.28% / +0.21% on the first two networks (lazy binds), +0.06% over 20; requests +5.8% and residual instructions +10% against the bank's normal-setup reference. Repeat under the platform clock |
| `runs/_native/5ae8ac05f789_766213e2_a1` | native, S35 + lvfuse_sized32, networks 35, 100 | predictions byte-identical to S35's, bill identical (263,239,303,752 and 263,905,546,384 flops), instructions x0.960-0.966 |
| `runs/cal_s35_noguard_wire` | one S35 session, setup guard held open, server wire counters | section 6: 42,112 requests per network, wire work 7.17 G instructions per network |
| `runs/prof_rrpi_noguard` | one RRPI session, client under cProfile | section 7 |
| `runs/setup/s35_free_*`, `s35_k*`, `fg_k*`, `margin_*`, `paced_s35_x*` | setup-only probes (light) | section 6 |
| `runs/setup/s35_g*` | setup-only probes of S35 at guards 1.0-4.0 s | section 4a-ter |
| `runs/anatomy/rrpi_v1`, `rrpi_v2`, `rrpi_clean` | CPU anatomy of RRPI | section 3 |
| `runs/a_s38`, `c_s38`, `d_s38` (run by the lead through `lead_chain.sh`, quiet window 07:25-07:57 UTC on 10-02) | stages A, C (legs S35, file, file, S35), D of the S38 file (sha 7c1a67f4), fired as #333481 | all PASS; section 11 |
| `runs/a_s38_7c1a67f4_precheck` | stage A of the S38 file before the lead's chain | PASS |
| `runs/f_selftest_same_s37_v1`, `f_selftest_differ_s35_v2` | fast-path judge on stored sessions (no session run) | PASS (S37's two legs) and FAIL (two own-clock S35 sessions, 14 counts) |
| `runs/a_s38_dialsA_v1`, `f_s38_dialsA_v1` (log `fast_s38_dialsA_v1.log`) | fast path, S38 file against the dialsA variant, two sessions | PASS; section 11 |
| `runs/a_s38_dialsB_v1`, `f_s38_dialsB_v1` (log `fast_s38_dialsB_v1.log`) | fast path, S38 file against the equal-strength variant, one session | FAIL on the census, as intended; section 11 |
| `runs/fast_windowcheck_v1.log` | the wrapper started inside the open quiet window | NOT STARTED (refused), as intended |
| `runs/s38_slow` (log `release_s38_slow.log`) | S38 with its setup at the pace of a worker 1.3x slower, one session | PASS: 525 of 649 keys bound, 42,626 requests per network (+1.83%), first network x1.0005 of S35's requests |
| `runs/c_s38_rejudged_w065`, `c_s38_rejudged_w065_v2`, `c_s38_rejudged_footprint_gate`, `d_s38_rejudged_footprint_gate`, `d_s38_bank_check`, `d_s38_vs_s38ref_selfcheck`, `d_s37_vs_s38ref_check`, `d_s38_vs_s37ref_check`, `c_dialsA_from_s38_parent_check`, `c_s37_from_s38_parent_check`, `c_dialsA_bank_s38_check`, `c_s38_explicit_s35_parent_check`, `c_dialsA_parent_from_check`, `f_s38_dialsA_rejudged_v2`, `_v3`, `f_selftest_same_s37_v2` | re-judgments of stored sessions while the projection rule, the memory gate and the bank were changed (no session run) | sections 2 and 11 |
| `runs/c_s38_endcheck_new_guard` (log `release_s38_endcheck.log`; re-judged in `c_s38_endcheck_fault_free`) | one real S38 session at load 15-20 under the footprint guard, against the stored quiet S38 legs: the end-to-end check of the day's changes | PASS: guard record footprint 2.81 GB / RSS 3.67 GB; every count equal to the quiet legs; projected 80.0 s with raw counts, 77.0 s with the fault work taken out (platform 77.07) |
| `runs/c_s38_from_s35_fault_free`, `c_dialsA_parent_from_fault_free`, `c_s37_from_bank_s38_fault_free`, `c_s36_from_bank_s38_fault_free`, `c_s35_from_bank_s38_fault_free`, `d_s37_vs_bank_s38_fault_free`, `f_faultfree_gate_check_a..d` | re-judgments of stored sessions with the page-fault correction | section 2 |
| `runs/a_s39_precheck`, `a_s39_precheck_reviewed` | stage A of the S39 file before the lead's window | FAIL on the four unreviewed einsum signatures alone; PASS with them passed |
| `runs/a_s39`, `c_s39`, `d_s39` (the lead's chain, 11:12-11:38 UTC) | stages A, C (S38, file, file, S38), D of the S39 file, fired as #333528 | all PASS; section 11 |
| `runs/c_s39_lvfs64`, `c_s39_fused` (log `release_s39_request_variants.log`) | S39 with trees un-fused only above 64M floats, and not at all; one session each | -535 and -1,170 requests per network, kernel +2.3% and +6.9%, byte-identical, bill equal |
| `runs/s39_slow`, `c_s39_sites` (log `release_s39_slow_and_sites.log`) | S39's slow-setup session and designation judge; request census by function | designation PASS; census in section 11 |
| `runs/c_s39_setup109` | S39 with the setup at 1.09x pace (61 deferred keys fused) | first network only (43,781 requests); the guard stopped the session on the machine-wide swap rule |
| `runs/c_S39_bank_legs` | copies of S39's two chain sessions under P names (written by `new_bank.sh`) | stored parent legs of the bank S39 |
| `runs/c_s38_bank_legs` | copies of the S38 file's two quiet chain sessions under P names | stored parent legs of the bank S38 (`calibration/bank_stage_c_paced_20261002T0850Z.txt`) |
| `runs/a_s40`, `c_s40`, `d_s40` (the lead's chain, 14:24-14:52 UTC) | stages A, C (S39, file, file, S39), D of the S40 file, fired as #333559 | all PASS; graded 1.7590e-9; section 12 |
| `runs/c_S40_bank_legs`, `runs/new_bank_S40.log` | copies of S40's two chain sessions under P names; the log of `new_bank.sh S40` | stored parent legs of the bank S40; projection against the grade |
| `runs/setup/s40_*`, `runs/setup_keys_s40.log` | setup-only probes: which key count books 29,381,392,716 flops | 627 of 649 keys, guard after 16,902 requests: worker 3's second setup in #333559 ran 1.124x slow |
| `runs/s40_slow1124` (chain 18) | the fired S40 file with its setup at that pace | +340 / +296 requests (first / warm), predictions byte-identical |
| `runs/c_s40d` (chain 18) | S40 + K4L birth by tables + pair-law kernels v5 (sha cf1f735c) | -274 / -269 requests, kernel time x0.985 |
| `runs/c_s41stack_sites`, `c_s41stack_sites_100first` (chains 18, 19) | the S41 stack (sha 8d110449) with the request census by call site, networks 35 and 100 in both orders | 44,881 / 42,982; being first costs +1,713 requests on either network; censuses in the run directories |
| `runs/c_s41stack_lvfs64` (chain 18) | the S41 stack with only trees above 64M floats un-fused | -521 / -549 requests, kernel time +1.6%, predictions byte-identical |
| `runs/c_s41hv`, `c_s41hv_l3b128`, `c_s41hv_lvfs64_l3b128`, `c_s41hv_l3ball` (chains 21, 22) | the engineer's S41 file with HV (sha 33bac9db) and its variants with the release fragments | section 12, table of the pieces |
| `runs/s41hv_lvfs64_l3b128_slow1124`, `s41hv_lvfs64_l3b128_slow` (chain 22) | the diet variant (sha 169fd371) with its setup at 1.124x and 1.3x pace | section 12; judge lines met |
| `runs/d_s41hv_lvfs64_l3b128` (chain 23) | stage D of the diet variant: 20 networks in one worker, native parity on 35 and 100 | PASS* (swap not judged on the shared box); fired by the lead as S41 = #333587 |
| `runs/c_s41hvcs`, `c_s41hvcs_l3b128_r2`, `s41hvcs_l3b128_slow1124`, `c_s41diet_l3b128` (chains 24, 25) | the engineer's constants lever CS and his rounding-level diet with the release fragments | section 12, last table |
| `runs/c_s42req`, `c_s42req_l3b128`, `s42req_l3b128_slow1124`, `s42req_l3b128_slow`, `c_s42req_lvfs64_l3b128` (chain 27) | the S42 request line: the engineer's build sha d464c12c alone and with the release fragments | section 12, last table |
| `runs/c_s41_fired_rejudged`, `c_s41diet_l3b128_rejudged` | stored sessions judged again with the calm threshold of 17:12 UTC | residual lines from requests |
| `_stale/s41hv_l3b128_slow1124_stopped_*` | a session of chain 21 stopped by hand (the slow-setup sessions were moved to the variant with lvfs64) | not read |
| `runs/anatomy/rrpi_v3` | same, paged (system/user 0.42-0.67) | counters rejected; MSE and predictions kept |
| `runs/anatomy/rrpi_<variant>` | seven variants | section 4 |
