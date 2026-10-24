# Release runbook: qualifying a candidate estimator before the lead submits it

Lane: `O/release/` (O = `P2/opus_reexam_20261001_v1`). Written 2026-10-01 by the release / wall-time engineer; this version
2026-10-02 09:00 UTC (previous: `_stale/RUNBOOK_20261002T0840Z_before_bank_s38_and_footprint_gate.md`,
`_stale/RUNBOOK_20261001T2120Z.md`). Changes of 2026-10-02: fast path (07:40), three-fire calibration (08:40), memory gate on
footprint and the bank re-based on S38 (08:50-09:00).
Nothing in this lane uploads or submits. The pipeline ends with a packaged, validated archive and its verdict lines; the upload
(`whest submit`) is the lead's step and is not in any script here. No script reads the API key.

## One command

```
bash O/release/quick.sh   <candidate.py> <tag> ["reviewed einsum sig;sig"]         # 12 min quiet, 30-40 min on a loaded box
bash O/release/qualify.sh <candidate.py> <tag> [parent.py] [stage-B gate options]  # everything; hours on a loaded box
```

```
bash O/release/designation_test.sh <file.py> <tag> ["reviewed einsum sig;sig"]     # about 35 min, only in an announced quiet window
```

Since 2026-10-10 (final week), a new physics file from the engineer is qualified as a BANK / SAFE pair in one run:

```
[ONLY=safe|bank] [SAFE_FRAGS=...] [DRY=1] bash O/release/qualify_pair.sh <engineer file.py> <tag> ["reviewed sig;sig"]
```

bank = file + `Rl3_bind128`; safe = file + SAFE_FRAGS (default `l3_bind128 lvfs_limit_off sym_two_strips`); stage A of both,
then stage C (platform setup clock, constants charged, parent = the stored legs of the bank in force), stage D, the 1.3x
slow-setup session and `designation_judge.py` for the bank ('BANK VERDICT' in `runs/qualify_<tag>.log`, about 45 minutes on a
quiet box), then the same for the safe file. DRY=1 stops after the builds and stage A (seconds). Signatures already reviewed
for S43 (10-10 02:57Z): `alsqj,lsjk,lk->aqsk;a,sq,k->aqsk` (F4Cv2's birth), `kn,kc->nc` (final-layer columns).

`designation_test.sh` is the test a file must pass before it is designated for the private run (rule accepted by the lead on
2026-10-02): the pair S35, file, file, S35 in one window under the platform clock, one session with the setup at the pace of a
worker 1.3x slower, the 20-network worker with native parity; `designation_judge.py <tag>` then prints one line per criterion
(the rule as of 2026-10-02 18:30 UTC: at most 43,800 requests on the first network after the setup and at most 42,700 on
the warm one (stage C's networks 35 and 100), at most +2% under the slow setup, binding phase at most 18,600 requests with every
key bound, projected worst-network wall at most 100 s, handles under 70,000, server + client peak footprint under 7.5 GB,
outputs finite, bytes and bill equal to native; for a graded file also: no network lost, its worst network at most 0.36 s of
residual and 105 s of roundtrip in its own run. Means in seconds are printed and not judged: the platform's speed moves by
several percent from hour to hour, S41's run was 3-4% slower than S40's in everything) and one verdict:
PASS, FAIL, REPEAT (the machine spoiled a measurement) or INCOMPLETE. The limit a file meets first is the 0.4 s residual on a
worker's first network after a setup: judge a file by the two counts, not by their mean. On a loaded box read a session's
requests, bills and prediction bytes (exact at any load) and leave kernel time and the residual's instruction count to a quiet
window: a server that pages fails the stage's `clean` gate, and the guard stops a session at 256 MiB of added system swap
whoever caused it (repeat the session). Chains of sessions wait on a pid file or on the previous chain's CHAIN DONE line,
never on `pgrep -f <script name>` (a waiter written that way matches its own shell).

```
bash O/release/fast_path.sh <qualified.py> <new.py> <tag> [in-window] [--fresh] [--native Q.json N.json]   # 2-3 min per session quiet
```

`fast_path.sh` (2026-10-02) is for a file that differs from an already qualified file ONLY IN LITERALS (the trainer's dial
candidates): it replaces the hour of chain by one or two sessions and needs no quiet window, because it gates on counts.
`<qualified.py>` is the last file that went through the full chain (stages A, C, D on record under `runs/`). Steps:
`literal_diff.py` (2 s: the two sources are the same code and differ only in float literals, packed base64 arrays and
docstrings, each listed; integers, switches, short strings and code differences fail it), stage A of the new file against the
qualified file's einsum signatures (the archive to fire), then `fast_path.py`: one worker session of each file under the
platform clock on networks 35 and 100 (the qualified file's session is reused when a stored one holds the full census) and the
census compared for the setup and each network: requests, free trips, handles freed, peak handles, kernel calls per operation
name, numpy calls per function, billed calls and flops per operation family, bill, stored bytes by number type, setup requests
/ handles / keys bound / keys fused: all equal (a bill or byte count that moves by under 0.1% with every call count equal is
a moved row count: reported, passes); residual and kernel instructions within +-3% / +-2% when both sessions were calm,
+-20% / +-10% otherwise (a loaded session reads up to 13% / 6% above a quiet one of the same file). One line: `[fast path] PASS` (the new file inherits the qualified file's wall, residual, memory and
handle projections, printed from its stage C and D records), `SAME COST` (the first file has no stage C and D record: nothing
to inherit), `REPEAT` (every count equal, but two loaded sessions disagree in instruction counts beyond the tolerance: run it
again) or `FAIL` (full chain). A float's value can steer operations (S38: equal F4 strengths per age take a scalar path, a
K4L strength of exactly 1.0 skips a multiply), which is why the literal proof alone is not the verdict. It refuses to start
while the lead's quiet window is open unless called with `in-window`. `selftest_literal_diff.py` and `smoke_boundary.py` are
the self-tests of the proof and of the session instruments.

`quick.sh`: packaging, setup margin, two boundary sessions under the platform clock against the bank's STORED legs
(the newest `calibration/bank_stage_c_paced_<UTC stamp>.txt` names the stage-C run whose P legs are used; store new ones in
a calm window with `stage_c_boundary.py <S35> <tag> --short-setup --legs PP`: legs measured while the box crawls read 2-4%
high in kernel work and flatter every candidate judged against them), and the slow-setup stress session. It is
the right tool while candidates are still changing; the 20-network worker (stage D) and the native validation (stage B) are
then run once on the file that will be submitted.

runs the stages one after the other (ONE heavy process at a time, each launch behind `O/gate.sh`) and prints one line per stage:
`[A package]`, the setup-margin row, `[C boundary]`, `[C boundary, setup of a x1.3 slower worker]`, `[D serving]`, `[B native40]`. Each line
carries the measured numbers and the limit each was held to. Everything is written under `O/release/runs/<tag>_{a,c,c_slow,d,b}/`;
a tag can be used once (directories are never overwritten). Stages can also be run alone.

Three verdicts exist. `PASS`: every gate passed. `FAIL`: a gate failed. `PASS*` (stages C and D only): every gate that does not
depend on other people's use of the machine passed, and only `swap` (system-wide swap growth) or `setup` (a wall-clock time)
failed while the machine was shared (1-minute load above 3.0 before launch). Those two are then reported as NOT JUDGED and the
stage has to be repeated in a quiet window before a release. A `FAIL` naming only `clean` says nothing about the candidate: the
server paged during the measurement (see below) and the stage has to be repeated.

| stage | command | what it does | time (quiet machine) |
|---|---|---|---|
| A | `python -B stage_a_package.py <cand> <tag> [--reviewed "sig;sig"]` | `whest validate` (contract), `whest package`, `whest validate-package` into `runs/<tag>/`; archive holds exactly `estimator.py` (byte-identical to the candidate) + `manifest.json`; source has no numpy import; every einsum signature already occurs in the bank. New signatures are printed with their operands and line and fail the stage until someone has read them and passes them in `--reviewed`; a new einsum call that passes the same expression twice fails outright (flopscope infers a symmetric output from a repeated operand and validates it in float32: it can raise on single networks) | 5 s |
| setup margin | `python -B setup_margin.py name=<cand>` | one light setup-only probe (under 1 GB): requests of the setup's binding phase against what a platform worker gets through before the estimator's 4.0 s guard; keys bound by a normal worker, by one 6% slower and by one 30% slower | 10 s |
| C, two sessions | `python -B stage_c_boundary.py <cand> <tag> --short-setup --legs CC --parent-from runs/c_s38_bank_legs` | the same projection against the bank's STORED legs (S38's two sessions of its chain); as good as the pair since the page-fault work is taken out of the counts | 4 min |
| C | `python -B stage_c_boundary.py <cand> <tag> --short-setup` | production path (numpy-free client -> flopscope server), paired with the bank in the order parent, candidate, candidate, parent; setup (under the platform's clock) + networks 35 (cold) and 100 per session; requests, kernel work, residual, memory; projection onto the platform | 6 min |
| C stress | `... <tag>_slow --setup-slowdown 1.3 --legs C --parent-from runs/<tag>` | one candidate session whose setup runs at the pace of a worker 1.3x slower (the re-set-up worker of the failed #333099); prints how much slower per request a worker with such a setup may be before the 0.4 s and 120 s limits are reached, and what the first network after that setup costs | 2 min |
| D | `python -B stage_d_serving.py <cand> <tag> --short-setup` | ONE worker session: setup + 20 networks back to back, server handle cap 70,000; finite outputs, peak handles, memory; native parity on networks 35 and 100 (bytes and bill); projection against the bank's stored 20-network reference | 13 min |
| B | `python -B stage_b_native40.py <cand> <tag> --parent <parent> [--set fresh40]` | native (numpy/OpenBLAS, 1 thread) 40 networks, both arms, same rank lists, same order, fresh process per 10 networks, arms alternating; pooled MSE ratio, bill ratio, adjusted ratio, wins, bootstrap interval, worst network, CPU ratio | 35 min (17 min when the parent is stored) |

Python for all: `/Users/matt/arc-white-box-estimation/.venv/bin/python` (the scripts pick the server, client and native
interpreters themselves: `common.py`).

## Limits each stage is held to (the ones used for S35)

| quantity | limit | stage | measured how |
|---|---|---|---|
| contract, package, archive contents, static source checks | all pass | A | `whest`, tar inspection, AST |
| adjusted score vs parent, 40 networks | <= -1% (bank gate; override per purpose) | B | mean of MSE x max(0.1, bill/2^41) |
| bootstrap 95% interval of the adjusted change | upper end < 0 | B | 20,000 resamples of networks |
| wins | >= 28 of 40 | B | networks with lower adjusted |
| worst single network | <= +10% | B | |
| peak footprint, server + client (gate `memory`) | < 7.5 GB | C, D | the two processes' lifetime peak footprints as the kernel reports them, summed; the guard (`guard_footprint.py`) also kills a session whose group footprint reaches 7.5 GB, 20 samples/s |
| group RSS | backstop 10 GB (a kill, not a gate) | every session | guard; on this machine RSS counts freed pages that are reclaimed only under memory pressure (6.0-6.8 GB in a quiet window, 2.9-3.9 GB loaded, at the same 2.1-2.7 GB footprint), so it is not compared with the platform's 8 GB (lead's decision 2026-10-02 08:37 UTC) |
| added system swap | 0 | C, D | guard |
| setup: every tree key bound before the guard (`binding`) | all keys, at a normal platform worker's pace | C, D with `--short-setup` | the estimator's own setup log under the platform clock |
| setup did not fall back (`setup_ok`) | no "restored after failure" in the setup log | C, D | |
| setup() wall (`setup`) | <= 4.1 s | C, D without `--short-setup` | client clock; only meaningful in a quiet window |
| projected residual, worst network | <= 0.4 s | C, D | bank's platform residual x candidate/bank ratio of residual instructions |
| projected wall, worst network | <= 105 s | C, D | bank's platform kernel, overhead, residual scaled by measured ratios, x 1.151 |
| survivable slowdown of a worker (printed, not gated) | S35: x1.18 residual, x1.26 wall under a normal setup | C, C stress | 0.4 s / projected residual max and 120 s / projected worst wall, under a normal setup and under a x1.3-slow setup |
| peak array handles in one worker over 20 networks | < 70,000 | D | counted in the server; the server refuses handle 70,001 |
| outputs | all 20 x 16,384 finite | D | |
| native = production | bytes and bill identical on networks 35, 100 | D | |

For a CPU variant that must keep the predictions, run stage B as
`--max-adjusted-pct 2 --max-raw-pct 0.05 --min-wins 0 --ci-upper 100 --max-regression-pct 10` (bill at most +2%, pooled MSE at
most +0.05%). For anything that touches learned parameters or rounding use `--set fresh40` (40 newer ranks nobody has trained or
selected on): on the historical valid40 every perturbation of the estimator reads slightly worse than it is (trainer, 21:50 UTC).

## When a file has been graded

`bash O/release/new_bank.sh <name> <file.py> <eval_<id>.json> <chain tag>` makes it the bank (only a file graded with no network
lost, and only after its full chain: the script needs `runs/c_<tag>` and `runs/d_<tag>`): it prints the projection against
the platform record (`check_projection.py`), writes `calibration/bank.json` (`set_bank.py`), installs its worker run as stage
D's reference and stores its chain legs as the parent legs of `quick.sh`. Then the designation test of the new bank needs one
session outside any quiet window: `python -B stage_c_boundary.py <file> <tag>_slow --setup-slowdown 1.3 --legs C --parent-from
runs/c_<tag>` and `python -B designation_judge.py <tag> --pair runs/c_<tag> --slow runs/<tag>_slow --worker runs/d_<tag>`
(the judge reads the platform record of a graded file by itself, through the stage-A archive and the lead's fire receipt).
A file qualified by the fast path has no chain of its own and does not become the bank: the bank stays the file it was
compared with.

A bank whose worker run was made on a loaded box (S41: fired without a quiet window, stage D at load 12-15) owes that run
again in a quiet window before it is designated: `python -B stage_d_serving.py <file> d_<tag>_quiet --short-setup`, then
`designation_judge.py <tag> --pair runs/c_<tag> --slow runs/<tag>_slow --worker runs/d_<tag>_quiet`, and the clean run
becomes stage D's reference: `python -B make_reference.py runs/d_<tag>_quiet/worker <eval json>` and copy its output over
`calibration/bank_stage_d_paced.json` (the replaced one into `_stale/`). A graded run's means in seconds are that hour's:
to tell the hour from the file, split the run's networks by position and compare kernel time per flop, overhead per round
trip and residual per request with the previous bank's run (RESULT.md section 12, S41).

After a grade, three tools read the record and the risk (seconds, no sessions): `python -B platform_networks.py <eval json>`
(first networks against the others, kernel time per flop of the worst networks, the flops every setup booked; a setup that
booked fewer flops stopped binding early: `python -B setup_keys.py <file> <tag> <flops>` names the key count and the pace of
that worker); `python -B loss_risk.py name=<eval json> name=<requests> ...` (share of the analyst's 729 public runs whose worst
network went further above their team's usual than the file has room); `python -B variant_table.py <reference run> <runs>`
(first / warm requests, setup, bill, kernel work and byte parity of variant sessions against a reference session).
What a first network costs function by function: two `--sites` sessions with `--ranks 35,100` and `--ranks 100,35`, then
`python -B first_network_cost.py <run>/C0 <run>/C0`; the census of one session: `census_by_function.py <session> [<other>]`.

## How the projection works and what it can be trusted for

The platform reports, per network: kernel (server numpy time), overhead (flopscope machinery and wire), residual (the
participant's own Python outside flopscope calls; limit 0.4 s), and their sum, the predict wall (limit 120 s for the round trip).
A candidate is projected by scaling each part of THE BANK's platform record. The bank is the newest graded file
(`calibration/bank.json`, written by `set_bank.py <name> <file> <eval json> <its stage_c.json>`; `common.py` reads it): since
2026-10-02 08:50 UTC S38 (#333481, 100 networks: kernel 68.36 s, overhead 8.42 s, residual 0.294 mean / 0.332 max, wall
77.07 mean / 88.27 max); before that S35 (#333111: kernel 73.41 s, overhead 8.75 s, residual 0.304 / 0.339, wall 82.47 /
94.93). Stage C runs the bank's legs beside the candidate's (`--legs PCCP`), stage A takes the bank's einsum signatures as
platform-proven, stage D projects from the bank's stored worker run. The nearest graded parent extrapolates least: S35 pays
0.183 ms of overhead per round trip on the platform, S36 / S37 / S38 0.177-0.179, and the share of memory-bound work in the
kernel time is parent-specific (S35 0.65, S38 0.60). Another parent on request: `--parent <file> --platform-record <eval json>`
(S35: `H/eval_333111.json`). The parts:

- kernel x (candidate / bank kernel work on the same local networks),
- overhead x (candidate / bank round trips: requests + hidden free trips),
- residual x (candidate / bank residual instructions),
- worst network = projected mean wall x the bank's own max / mean on the platform (S38 1.145; S35 1.151).

Work is counted in CPU instructions and cycles read from the macOS kernel (`rusage.py`), not in seconds, because this shared
machine's seconds are not reproducible: identical single-thread work swings 2-2.7x within seconds while other lanes and the
GPU job run (the performance cores are clocked down in bursts; measured 2026-10-01, see RESULT.md). Kernel time is the PROXY
`sum over operation types of instructions x best-seen cycles per instruction`; memory-bound and CPU-bound operation types are
scaled separately because the platform is slower on the former (`project.py`). **The work of page faults is taken out of the
instruction counts** (`project.fault_free`, since 2026-10-02 09:20 UTC): when the machine is short of free memory the kernel
pages every large array of the server in again and that work is counted, 14,200 instructions per minor fault (8,900 in the
memory-bound operations, 5,300 in the CPU-bound ones; fit over 42 stored session-networks). Raw counts of one file read 2-10%
apart between a quiet and a crawling box; the fault-free ones repeat to 0.27% rms (worst 0.8%), so legs of different windows
and stored legs can be compared at any load. They are still not proof against real paging: a network during which the server
paged (system/user CPU above 0.30; clean is 0.10-0.26) decompresses pages at a far higher price per fault. Such network measurements are dropped from both arms; stage C fails on gate `clean` only if some
network has no clean measurement left in an arm, stage D if fewer than half of the networks are clean in both runs. Seconds are
printed as a cross-check and mean something only when the lead has paused the other lanes.

Three builds have been fired with a projection (S36, S37, S38; RESULT.md section 2): the residual was right to 1.5%, 0.5% and
0.0%, the mean wall as fired to -1.1, -3.7 and -2.1 s. Two things were wrong and are corrected: page-fault work in the counts
(above: it made the answer depend on which window each arm was measured in) and the memory-bound share of the kernel
projection (0.72, now 0.65). Re-projected from their stored legs with both corrections, whichever window the S35 legs come from,
the kernel time of the three reads +1.4..+2.1%, -0.6..+0.1% and -1.9..-1.1% (the platform repeats a run's kernel time to about
2%). Read a projected mean wall as +-1.5 s. Stage D under the platform clock projects as well, from S38's own 20-network worker
run and platform record (`calibration/bank_stage_d_paced.json`, built by `make_reference.py runs/d_s38/worker
<eval_333481.json>`): S37's worker run of another window projects 78.7 s from it against 78.0 measured. When a newer file is graded, build its reference the same way and copy it over `bank_stage_d_paced.json` (keep the
old file: the name carries the sha).

Calibration and its error are in `RESULT.md` (section "Calibration"). Read that before trusting a projection for a candidate
whose kernel differs in kind from the bank's (different leaf sizes, strided views): for such a candidate ask the lead for a
quiet window and judge on paired seconds as well.

## The setup runs under the platform's clock (`--short-setup`, `setup_clock.py`)

The estimator's setup works against its own 4.0 s wall guard: it binds tree keys most-used first, then fuses them, and whatever
it did not bind is bound inside predictions (billed, more requests, never fused). On this machine the same setup gets through
146 to 649 keys depending on the load of the moment, so sessions were not comparable. The platform is pinned by the flops its
setups booked (deterministic per key): a normal worker gets through about 19,000 requests in the guard (range 18,600-20,100);
worker w-2 of #333111 managed 17,954 (623 of 649 keys); the slow re-set-up worker of #333099 14,589 (580 of 649 keys).
`--short-setup` replaces the estimator's setup clock by `requests so far x 4.0 s / 19,000`; `--setup-slowdown X` by the same
for a worker X times slower. The candidate file is not changed; only its module's clock name is rebound in the harness.
Consequences:

- how far a session's setup gets is the same in every session, whatever the load, and is that of a platform worker;
- the wall-clock setup gate is replaced by `binding` (all keys bound at a normal worker's pace);
- the bank's binding phase needs 18,555 of the 19,000 requests (97.7%): a candidate that lengthens it will not finish it on
  normal workers. `setup_margin.py` prints this for any candidate in seconds.

## Options worth knowing

| option | stage | what it does |
|---|---|---|
| `--short-setup` | C, D | the setup runs under the platform's clock at a normal worker's pace (see above). Use it for every qualification. |
| `--setup-slowdown X` | C, D | the same for a worker X times slower in setup: a stress run; with `--parent-from` a normal-pace run the line also says what the 0.4 s and 120 s limits look like on that worker. 0 = no guard (everything bound and fused). |
| `--parent-from DIR --legs CC` | C | takes the parent arm from the P legs of an earlier stage-C run instead of running it again (counts repeat to +-0.5% between clean sessions); halves the machine time per candidate |
| `--from-run DIR [--skip i,j]` | C | re-assesses finished sessions (after a change of the projection, or to leave out a leg that ran while the machine was overloaded) |
| `--from-worker DIR` | D | re-assesses a finished 20-network worker session |
| `--parent FILE --platform-record EVAL.json` | C | project from another graded file instead of S35: its source and the stored evaluation JSON of its submission (S38: `O/fire/s38_20261002/eval_333481.json`). The nearest graded file is the best parent; its legs must still be measured in the candidate's window (`--legs PCCP`) |
| `--reference FILE` | D | bank reference to project against; default: the stored one whose setup ran under the same clock (`calibration/bank_stage_d.json`, `bank_stage_d_paced.json`) |
| `--profile` | C | every prediction runs under cProfile in the client (`residual_report.py`); such a run is not for a verdict |
| `--set fresh40`, `--ranks`, `--max-adjusted-pct`, `--max-raw-pct`, `--ci-upper`, `--min-wins`, `--max-regression-pct` | B | network set and gate; defaults are valid40 and the bank gate |

Other tools: `setup_probe.sh <candidate.py> <tag> [guard seconds] [slowdown]` (setup alone at the boundary: wall, requests, billed
flops, keys bound; light) · `setup_keys.py <candidate.py> <tag> <flops>` (how many keys a platform worker had bound, from the
flops its setup booked) · `setup_phases.py runs/setup/<tag>.json` (requests and wire work per setup phase) ·
`compare_sessions.py <session A> <session B>` (per-network ratios of bill, requests, residual between two worker sessions) ·
`anatomy_run.sh <candidate.py> <tag> [ranks] [reference dir]` then `variant_eval.py <parent dir> <variant dirs...>` (bill,
prediction bytes, MSE and kernel CPU by operation in 2.5 minutes for two networks) and `anatomy_report.py <dir>` (CPU by
operation, phase, stage, layer, call site) · `residual_report.py <profile.pstats>` (where the client's own Python goes) ·
`fuse_bench.py L r c` (numpy-only cost of fused against single level operations) · `build_variant.py <parent.py> <name>
<fragment...>` (parent + fragments from `variants/fragments/`) · `own_gate.sh` (the lead's gate plus: swap flat for 30 s and
4 GB of free + file-backed memory; every stage launch goes through it).

## Things that go wrong, and what the scripts do about them

- **Starting anything.** Read the newest `[lead]` lines of `O/TEAM_LOG.md` first; the lead's messages reach a lane only when
  its turn ends. Do not arm a queue to start by itself on someone else's marker: on 2026-10-02 such a queue started in the
  minute the lead ruled out boundary sessions and ran 26 minutes against the rule.
- **Two people queueing in one lane.** Queue scripts wait on a marker file or on a line in a log, never on `pgrep -f <script
  name>`: a waiting launcher whose own command line contains the name blocks itself, and a name-based `pkill` hits every lane's
  script of that name (the engineer lane also has `chainNN.sh`). Stop a script by its pid. Release scripts are named
  `release_chainNN.sh` from 2026-10-02 on; the lead's are `lead_*.sh`.
- **Hang guard.** Sessions are stopped after 600 s + 600 s per network (stage C), 900 s + 600 s per network (stage D) and
  300 s + 400 s per network (native batches). These are hang guards, not speed limits: at load 25-30 a network takes 4-8
  minutes instead of 40-70 s.

- **Loaded machine.** Seconds inflate 2-10x (a setup that takes 4 s on the platform took 56-78 s here at load 25-35). Counters
  and, with `--short-setup`, the setup's reach do not move; only the run takes longer. Without `--short-setup` the setup stops
  early and the session measures a degraded worker: do not use such a session for a verdict.
- **Memory.** The machine keeps about 24 GB compressed and 15 GB wired; a server session needs about 6 GB. `gate.sh` is called
  before every session, and only ONE heavy release job may run at a time: two at once pushed the box into paging on 2026-10-01
  (19:27-19:37Z), which cost a trainer worker and contaminated three networks of the bank's own stage-D run. RSS under-reads
  when the compressor takes idle pages and over-reads freed-but-resident pages; the server's peak physical footprint is printed
  next to it.
- **Bills depend on cache history.** Arms are only compared on the same rank list in the same order in fresh processes; the
  native store (`runs/_native/<sha12>_<rank-list hash>_a<attempt>/`) is keyed by exactly that.
- **A setup that did not bind every key** bills the rest inside predictions, so the worker's bill differs from the native bill
  and `native_parity` fails although the bytes are identical; the stage-D line prints bytes and bill separately.
- **A stage dies.** Guard records and logs stay in the run directory; nothing is retried automatically and nothing is deleted.
  Use a new tag for the rerun.

## Files

`common.py` paths and limits · `rusage.py` kernel counters · `server_light.py` stock server + handle peak + per-operation kernel
counters + wire work · `boundary_client.py` numpy-free worker · `boundary_run.py` one worker session · `setup_clock.py` the
platform clock for the setup and its calibration · `project.py` projection · `stage_{a,b,c,d}_*.py` · `native_probe.py` native
bill/MSE · `cpu_anatomy.py` + `anatomy_report.py` CPU by call site · `qualify.sh` all stages · `quick.sh` the quick path · `designation_test.sh` + `designation_judge.py` the designation test ·
`check_projection.py` platform record against the projection after a fire · `calibrate.py` backtest of the
projection over the banks with platform pages · `make_reference.py` bank reference for stage D · `build_variant.py` +
`variants/fragments/` + `variant_eval.py` + `anatomy_run.sh` CPU variants · `setup_probe.sh` + `setup_probe_client.py` +
`setup_margin.py` + `setup_keys.py` + `setup_phases.py` setup-only probes · `compare_sessions.py` · `residual_report.py` ·
`platform_networks.py` a graded run network by network · `loss_risk.py` loss probability of a fresh run · `variant_table.py` · `first_network_cost.py` + `census_by_function.py` request censuses · `variants/fragments/l3_bind128.py`, `lvfs_limit64.py` the request diet as release fragments (builder levers `Rl3_bind128`, `Rlvfs_limit64`) ·
`calibration/` stored bank references · `RESULT.md` what ran and what it showed.
