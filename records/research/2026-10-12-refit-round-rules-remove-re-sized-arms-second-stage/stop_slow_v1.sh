#!/bin/bash
# TAG opus-trainer RAD (10-12 16:4xZ; lead 16:5xZ, radlab's STOP/SLOW rule, pre-registered in TEAM_LOG before S52's check): follows the rounds
# S52..S66 in order; a round counts once it is over (run_RAD_S<N>_gen/STOP written and no S<N> chain alive). A finished round is DRY when its
# freeze was NULL or its LOO is above -0.03% (cert/RAD/S<N>/FROZEN_POINT_v1.json), or it froze nothing; two DRY rounds in a row -> the line
# drops to 2 workers (cert/RAD/WORKER_CAP = 2; the old file archived; maintain_workers_v5 applies it within minutes) and the CPU is free for
# error-side work; one TEAM_LOG line per finished round and one at the trigger. Ends after the trigger or at 10-17 12:00Z.
# usage (from O/trainer): nohup bash cert/RAD/stop_slow_v1.sh > cert/RAD/stop_slow_v1.log 2>&1 &
cd "$(dirname "$0")/../.."
O=..; PY=/Users/matt/anaconda3/bin/python; END=$(date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "2026-10-17T12:00:00Z" +%s)
log() { echo "[trainer] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> $O/TEAM_LOG.md; echo "$(date -u +%H:%M:%SZ) $*"; }
over() { [ -e run_RAD_S$1_gen/STOP ] && [ $(ps -axo command= | awk -v n=$1 '$1 == "bash" && $2 ~ ("^cert/RAD/S" n "/chain")' | wc -l) -eq 0 ]; }
dry=0
for n in 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66; do
  until over $n || [ $(date +%s) -ge $END ]; do sleep 120; done
  [ $(date +%s) -ge $END ] && exit 0
  R=$($PY -c "
import json; from pathlib import Path
p = Path('cert/RAD/S$n/FROZEN_POINT_v1.json')
if not p.exists(): print('dry none'); raise SystemExit
f = json.loads(p.read_text())
print(('dry' if f['null'] or f['loo_pct'] > -0.03 else 'wet'), ('NULL' if f['null'] else 'LOO %+.3f%%' % f['loo_pct']))")
  set -- $R
  if [ "$1" = "dry" ]; then dry=$((dry + 1)); else dry=0; fi
  log "S$n STOP/SLOW count (radlab's rule): ${R#* } -> $1, dry rounds in a row = $dry (2 -> the line drops to 2 workers)"
  if [ $dry -ge 2 ]; then
    [ -e cert/RAD/WORKER_CAP ] && mv cert/RAD/WORKER_CAP cert/RAD/WORKER_CAP.archived_$(date -u +%Y%m%dT%H%M%SZ)_stopslow
    echo 2 > cert/RAD/WORKER_CAP
    log "STOP/SLOW TRIGGERED after S$n (two dry rounds in a row): cert/RAD/WORKER_CAP = 2, the RAD line continues at 2 workers and the CPU is free for error-side work (lead / radlab 16:5xZ)."
    exit 0
  fi
done
