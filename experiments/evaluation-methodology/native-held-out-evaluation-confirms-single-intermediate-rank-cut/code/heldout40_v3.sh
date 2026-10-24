#!/usr/bin/env bash
# TAG fable-porter (PT-64d, lead 11:2x: S27s out; S27t = L10-only in): RESUMABLE held-out runner on the analyst's 40 held-out TRAIN640 nets, arms S27 (fire copy),
# S27r (384/320/320), S27t (384/320/384); rows already in heldout40_3arm_rows.jsonl are reused (arm, rank); ONE process at a time (machine gate 6 GB),
# score_variant_t688.py, each net cold. Starts after S27t's VALID40 has taken the lock (S27t's build/gates keep memory priority); S27t nets need S27t's
# gates + smoke passed. Report heldout40_v3.json: per arm vs S27 VALID40 / held-out / pooled-80 (bootstrap), bar pooled adj <= -0.5% CI excl 0; + S27t vs S27r.
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md
NPY=/Users/matt/arc-white-box-estimation/.venv/bin/python; SV=$H/score_variant_t688.py; RAW=$H/heldout40_3arm_rows.jsonl; OUT=$H/heldout40_v3.json
A=$P2/fable_fire_v1/fire_candS27/candS27_S26_capA_mem.py; B=$H/stack/stack_S27r_mem.py; C=$H/stack/stack_S27t_mem.py
[ -e $OUT ] && { echo "output exists"; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
done_() { python3 -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1])]; sys.exit(0 if any(r['arm']==sys.argv[2] and r['rank']==int(sys.argv[3]) and 'mse' in r for r in rows) else 1)" $RAW $1 $2 2>/dev/null; }
one() { done_ $1 $3 && return; bash $H/machine_gate.sh 6 "heldout-$1-$3" > /dev/null; nice -n 10 $NPY $SV $2 $3 2>/dev/null | grep '^{' | sed "s/^{/{\"arm\": \"$1\", /" >> $RAW; echo "$(date +%T) $1 net $3"; }
echo "$(date +%T) waiting for S27t's VALID40 to take the lock"
until grep -q "lock taken, VALID40 stack_S27t_mem start" $H/fire_chain_S27t.out 2>/dev/null || ! pgrep -f "bash fire_chain_S27[t].sh" >/dev/null; do sleep 20; done
NETS=$(python3 -c "import json; print(' '.join(map(str, json.load(open('$H/heldout40_ranks.json'))['ranks'])))")
for r in $NETS; do one S27 $A $r; one S27r $B $r; [ -s $C ] && grep -q "\[7\] VALID40" $H/fire_chain_S27t.out && one S27t $C $r; done
for r in $NETS; do [ -s $C ] && one S27t $C $r; done
echo "$(date +%T) waiting for the S27t VALID40 verdict"
until [ -s $H/gate_S27t_bar.json ] || ! pgrep -f "bash fire_chain_S27[t].sh" >/dev/null; do sleep 20; done
python3 - $RAW $H $OUT <<'PY'
import json, sys, numpy as np
from pathlib import Path
rows = [json.loads(l) for l in open(sys.argv[1])]; H = Path(sys.argv[2]); R = {}
for x in rows:
    if 'mse' in x: R.setdefault(x['rank'], {})[x['arm']] = x
V = {a: json.load(open(H / f'paired_valid40_stack_{a}_mem.json'))['results'] for a in ('S27', 'S27r', 'S27t') if (H / f'paired_valid40_stack_{a}_mem.json').exists()}
vn = sorted(int(k.split(':')[1]) for k in V['S27'] if k.startswith('cand:'))
def adj(x): return x['mse'] * max(0.1, x['util'])
def stat(b, a, rb, ra):
    b, a, rb, ra = map(np.array, (b, a, rb, ra)); n = len(b)
    boot = np.random.default_rng(0).integers(0, n, (20000, n)); bc = b[boot].sum(1) / a[boot].sum(1) - 1
    return dict(n=n, adjusted=float(b.sum() / a.sum() - 1), ci95=[float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))], wins=int((b < a).sum()), raw=float(rb.sum() / ra.sum() - 1))
def pairs(arm, ref):
    ho = sorted(n for n in R if arm in R[n] and ref in R[n])
    vb = [adj(V[arm][f'cand:{n}']) for n in vn]; va = [adj(V[ref][f'cand:{n}']) for n in vn]; vrb = [V[arm][f'cand:{n}']['mse'] for n in vn]; vra = [V[ref][f'cand:{n}']['mse'] for n in vn]
    hb = [adj(R[n][arm]) for n in ho]; ha = [adj(R[n][ref]) for n in ho]; hrb = [R[n][arm]['mse'] for n in ho]; hra = [R[n][ref]['mse'] for n in ho]
    out = dict(valid40=stat(vb, va, vrb, vra), heldout=stat(hb, ha, hrb, hra), pooled=stat(vb + hb, va + ha, vrb + hrb, vra + hra))
    out['bar'] = 'PASS' if out['pooled']['adjusted'] <= -0.005 and out['pooled']['ci95'][1] < 0 else 'FAIL'; return out
S = {f'{a}_vs_S27': pairs(a, 'S27') for a in ('S27r', 'S27t') if a in V}
if 'S27r' in V and 'S27t' in V: S['S27t_vs_S27r'] = pairs('S27t', 'S27r')
json.dump(dict(tag='fable-porter PT-64d held-out v3', summary=S), open(sys.argv[3], 'w'), indent=1)
for k, v in S.items():
    print(k + ': ' + ' | '.join(f"{g} n{v[g]['n']} adj {100 * v[g]['adjusted']:+.3f}% [{100 * v[g]['ci95'][0]:+.3f}, {100 * v[g]['ci95'][1]:+.3f}] raw {100 * v[g]['raw']:+.3f}% wins {v[g]['wins']}" for g in ('valid40', 'heldout', 'pooled')) + f" | BAR {v['bar']}")
PY
echo "$(date '+%Y-%m-%d %H:%M') porter: NATIVE SCHEDULE EVIDENCE v3 (VALID40 + held-out TRAIN640 nets): $(python3 -c "import json; s=json.load(open('$OUT'))['summary']; print(' || '.join(k + f\": pooled n{v['pooled']['n']} adj {100*v['pooled']['adjusted']:+.3f}% [{100*v['pooled']['ci95'][0]:+.3f}, {100*v['pooled']['ci95'][1]:+.3f}] {v['bar']}\" for k, v in s.items()))") (heldout40_v3.json)." >> $TL
echo "$(date +%T) HELDOUT V3 DONE"
