#!/bin/bash
# TAG opus-trainer RAD (written 2026-10-13T16:2xZ; lead 16:2xZ: hold the raise until T146's final 24-network table (plus any (g) arm) has posted:
# T146 is today's fastest fire path and its 3 workers must not slow down; T145 was stopped by the lead 16:0xZ, its own final table may never post):
# = cap6_s56_v2.sh with the trigger 'the physicist's final T146 table (its anchored T146 AUTO TABLE line, written after its workers left, (g) included)
# AND scout-card's K24 poster gone after posting K24 tables' instead of T145's table. Baseline / compressor / throughput rules as v2.
# usage (from O/trainer): nohup bash cert/RAD/cap6_s56_v3.sh > cert/RAD/S56/cap6_s56_v3.log 2>&1 &
# TAG opus-trainer RAD (written 2026-10-13T16:0xZ; lead 16:0xZ: the box is memory-bound, so cap 6 only if it really speeds S56's evals): = cap6_s56_v1.sh +
# (a) the raise waits while top's PhysMem compressor is above 10 GB (and, as v1, while memory_pressure free < 50%);
# (b) a THROUGHPUT CHECK: S56's sizing evals per minute (result files by mtime) over the 20 min before the raise vs the 20 min after; if the rate after
#     is not at least 15% higher, back to 4 (retire_top_v2: indices 4 / 5 retire after their current eval) and one line; else one line and the cap stays;
# (c) a clean baseline: the raise comes no earlier than 20 min after both screens have finished (the physicist's final T145 table and scout-card's K24
#     poster gone), so the 20-min window before it has no screen workers in it and measures 4 workers alone.
# v1's other rules stand: back to 4 when S56's sizing is done (its SIZING DONE line or its run's STOP), or at once if memory_pressure free < 50% on two polls
# 30 s apart or swap grows > 256 MiB over its value at the raise.
cd "$(dirname "$0")/../.."
O=..; L=$O/TEAM_LOG.md; RUN=run_RAD_S56_gen
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
mp() { memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%'; }
swap() { sysctl -n vm.swapusage | sed -n 's/.*used = \([0-9]*\)\..*/\1/p'; }
comp() { top -l 1 -n 0 | sed -n 's/^PhysMem:.*, \([0-9.]*\)\([MG]\) compressor).*/\1 \2/p' | awk '{print ($2 == "G") ? int($1 * 1024) : int($1)}'; }
nres() { /usr/bin/python3 -c "import glob, os, sys; a, b = float(sys.argv[1]), float(sys.argv[2]); print(sum(1 for f in glob.glob('$RUN/results/a_RS56_*.json') if a <= os.stat(f).st_mtime < b))" $1 $2; }
done_sizing() { grep -qE '^\[trainer\] [^ ]+ S56 SIZING DONE' $L || [ -e $RUN/STOP ]; }
t146() { grep -qE '^\[physicist\] [^ ]+ T146 AUTO TABLE' $L; }
k24() { ! pgrep -f 'auto_post_k24_v1[.]sh' > /dev/null && grep -qE '^\[scout-card\] [^ ]+ AUTO TABLE K24' $L; }
back4() {      # $1 = reason
  R=cert/RAD/S56/retire_top_v2_cap6.log; [ -e $R ] && R=cert/RAD/S56/retire_top_v2_cap6_$(date -u +%H%M%S).log
  say "back to 4: $1"
  bash cert/RAD/retire_top_v2.sh S56 4 "lead 15:4xZ / 16:0xZ cap-6 window over: $1" > $R 2>&1
  say "retire_top_v2 done: $(tail -1 $R)"
}
say "armed: waiting for T146's final table and K24's poster to finish"
until t146 && k24; do
  done_sizing && { say "S56 sizing ended before both screens finished: cap stays $(cat cert/RAD/WORKER_CAP)"; exit 0; }
  sleep 30
done
E=$(date +%s); say "both screens done; clean 4-worker baseline from now, raise no earlier than $(date -u -r $((E + 1200)) +%H:%M:%SZ)"
while [ $(date +%s) -lt $((E + 1200)) ]; do done_sizing && { say "S56 sizing ended during the baseline: no raise"; exit 0; }; sleep 30; done
until [ "$(mp)" -ge 50 ] && [ "$(comp)" -le 10240 ]; do
  done_sizing && { say "S56 sizing ended while memory_pressure < 50% or the compressor > 10 GB: no raise"; exit 0; }
  sleep 30
done
T0=$(date +%s); B=$(nres $((T0 - 1200)) $T0); S0=$(swap); M0=$(mp); C0=$(comp); OLD=$(head -1 cert/RAD/WORKER_CAP | tr -dc 0-9)
cp cert/RAD/WORKER_CAP cert/RAD/WORKER_CAP.archived_$(date -u +%Y%m%dT%H%M%SZ)_cap$OLD
echo 6 > cert/RAD/WORKER_CAP
N=$(ls $RUN/results | grep -cE '^a_RS56_.*\.json$')
echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) S56 WORKER_CAP $OLD -> 6 (lead 15:4xZ / 16:0xZ / 16:2xZ; T146's final table and K24's tables are posted, then a clean 20-min 4-worker baseline): S56 sizing at $N of 512 evals, $B evals in the last 20 min at 4 workers; memory_pressure free $M0%, compressor ${C0}M, swap ${S0}M. Throughput check at +20 min: fewer than 1.15 x $B evals in the 20 min after the raise -> back to 4. Also back to 4 when S56's sizing is done, or at once if memory_pressure free < 50% on two polls 30 s apart or swap > $((S0 + 256))M; cert/RAD/cap6_s56_v3.sh." >> $L
say "cap $OLD -> 6 at swap ${S0}M, free $M0%, compressor ${C0}M, sizing $N/512, baseline $B evals / 20 min"
low=0; checked=0
while true; do
  sleep 30
  done_sizing && { back4 "S56 sizing done"; exit 0; }
  m=$(mp); s=$(swap)
  if [ "$m" -lt 50 ]; then low=$((low + 1)); else low=0; fi
  [ $low -ge 2 ] && { back4 "memory_pressure free ${m}% < 50% on two polls"; exit 0; }
  [ $((s - S0)) -gt 256 ] && { back4 "swap ${S0}M -> ${s}M (> 256 MiB)"; exit 0; }
  if [ $checked -eq 0 ] && [ $(date +%s) -ge $((T0 + 1200)) ]; then
    A=$(nres $T0 $((T0 + 1200))); checked=1
    if [ $((100 * A)) -lt $((115 * B)) ]; then
      back4 "throughput check: $A evals in the 20 min after the raise vs $B in the 20 min before (needs >= 1.15 x)"; exit 0
    fi
    echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) S56 CAP 6 KEPT (throughput check, lead 16:0xZ): $A evals in the 20 min after the raise vs $B in the 20 min before at 4 workers (x$(awk -v a=$A -v b=$B 'BEGIN {printf "%.2f", (b > 0) ? a / b : 0}')); compressor $(comp)M, memory_pressure free $(mp)%; back to 4 when S56's sizing is done." >> $L
    say "throughput check passed: $A vs $B"
  fi
done
