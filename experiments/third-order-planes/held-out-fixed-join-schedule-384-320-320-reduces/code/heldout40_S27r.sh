#!/usr/bin/env bash
# TAG fable-porter (PT-64, lead 09:5x): native held-out evidence for the 384/320/320 schedule. S27 (fire copy) and S27r (stack_S27r_mem.py) on the
# analyst's 40 held-out TRAIN640 nets (heldout40_ranks.json, from r320_features_ho40.jsonl), each net cold in its own process with the VALID40
# scorer (score_variant_t688.py = score_variant.py reading truth688). Starts after S27r's VALID40 verdict; takes the paired lock; 2 processes at a
# time (one per arm). Report: VALID40 alone (S27r vs S27 rows), held-out alone, and POOLED over the 80 nets with a 20000-resample net bootstrap;
# bar (lead): pooled adjusted <= -0.5%, CI excl 0. Output heldout40_S27r.json + TEAM_LOG line.
set -u
P2=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16; H=$P2/fable_porter_v1; TL=$P2/fable_refit_v1/TEAM_LOG.md
NPY=/Users/matt/arc-white-box-estimation/.venv/bin/python; SV=$H/score_variant_t688.py; A=$P2/fable_fire_v1/fire_candS27/candS27_S26_capA_mem.py; B=$H/stack/stack_S27r_mem.py
RAW=$H/heldout40_S27r_rows.jsonl; OUT=$H/heldout40_S27r.json
[ -e $RAW ] || [ -e $OUT ] && { echo "output exists"; exit 1; }
[ "$(shasum -a 256 $A | cut -c1-16)" = 9ff80a12de5a5fb8 ] || { echo "S27 fire copy sha changed"; exit 1; }
echo "$(date +%T) waiting for S27r's VALID40 verdict"
until [ -s $H/gate_S27r_bar.json ] || ! pgrep -f "bash fire_chain_S27[r].sh" >/dev/null; do sleep 20; done
[ -s $B ] && [ -s $H/paired_valid40_stack_S27r_mem.json ] || { echo "S27r file or its VALID40 rows missing"; exit 1; }
echo "S27r sha $(shasum -a 256 $B | cut -c1-16)"
bash $H/machine_gate.sh 12 heldout40-S27r
while ! mkdir /tmp/fable_paired.lock 2>/dev/null; do sleep 30; done
trap 'rmdir /tmp/fable_paired.lock 2>/dev/null' EXIT
echo "$(date +%T) lock taken"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for r in $(python3 -c "import json; print(' '.join(map(str, json.load(open('$H/heldout40_ranks.json'))['ranks'])))"); do
  nice -n 10 $NPY $SV $A $r 2>/dev/null | grep '^{' | sed 's/^{/{"arm": "S27", /' >> $RAW & a=$!
  nice -n 10 $NPY $SV $B $r 2>/dev/null | grep '^{' | sed 's/^{/{"arm": "S27r", /' >> $RAW & b=$!
  wait $a $b; echo "$(date +%T) net $r done"
done
python3 - $RAW $H/paired_valid40_stack_S27r_mem.json $H/paired_valid40_stack_S27_mem.json $OUT <<'PY'
import json, sys, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]; R = {}
for x in rows: R.setdefault(x['rank'], {})[x['arm']] = x
ho = sorted(n for n in R if all(a in R[n] and 'mse' in R[n][a] for a in ('S27', 'S27r')))
err = sorted(n for n in R if n not in ho)
V1, V0 = json.load(open(sys.argv[2]))['results'], json.load(open(sys.argv[3]))['results']
vn = sorted(int(k.split(':')[1]) for k in V1 if k.startswith('cand:'))
def adj(x): return x['mse'] * max(0.1, x['util'])
def arr(pairs): return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
ho_pairs = [(adj(R[n]['S27r']), adj(R[n]['S27'])) for n in ho]
v_pairs = [(adj(V1[f'cand:{n}']), adj(V0[f'cand:{n}'])) for n in vn]
def stat(pairs, raw_pairs):
    b, a = arr(pairs); rb, ra = arr(raw_pairs); n = len(b)
    boot = np.random.default_rng(0).integers(0, n, (20000, n)); bc = b[boot].sum(1) / a[boot].sum(1) - 1
    return dict(n=n, adjusted=float(b.sum() / a.sum() - 1), ci95=[float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))], wins=int((b < a).sum()), raw=float(rb.sum() / ra.sum() - 1))
S = dict(valid40=stat(v_pairs, [(V1[f'cand:{n}']['mse'], V0[f'cand:{n}']['mse']) for n in vn]),
         heldout40=stat(ho_pairs, [(R[n]['S27r']['mse'], R[n]['S27']['mse']) for n in ho]),
         pooled80=stat(v_pairs + ho_pairs, [(V1[f'cand:{n}']['mse'], V0[f'cand:{n}']['mse']) for n in vn] + [(R[n]['S27r']['mse'], R[n]['S27']['mse']) for n in ho]))
S['bar'] = dict(rule='pooled adjusted <= -0.5% with the CI excluding 0', verdict='PASS' if S['pooled80']['adjusted'] <= -0.005 and S['pooled80']['ci95'][1] < 0 else 'FAIL')
json.dump(dict(tag='fable-porter PT-64 S27r vs S27, VALID40 + 40 held-out TRAIN640 nets', errors=err, summary=S, heldout_rows={str(n): R[n] for n in sorted(R)}), open(sys.argv[4], 'w'), indent=1)
for k in ('valid40', 'heldout40', 'pooled80'):
    s = S[k]; print(f"{k}: n {s['n']} adjusted {100 * s['adjusted']:+.3f}% [{100 * s['ci95'][0]:+.3f}, {100 * s['ci95'][1]:+.3f}] wins {s['wins']}/{s['n']} raw {100 * s['raw']:+.3f}%")
print('BAR (pooled 80 adjusted <= -0.5%, CI excl 0):', S['bar']['verdict'], '| errors:', err)
PY
echo "$(date '+%Y-%m-%d %H:%M') porter: S27r HELD-OUT (native, 40 TRAIN640 nets, each cold; score_variant_t688.py) + VALID40, S27r vs S27: $(python3 -c "import json; s=json.load(open('$OUT'))['summary']; print(' | '.join(f\"{k} adj {100*s[k]['adjusted']:+.3f}% [{100*s[k]['ci95'][0]:+.3f}, {100*s[k]['ci95'][1]:+.3f}] raw {100*s[k]['raw']:+.3f}% wins {s[k]['wins']}/{s[k]['n']}\" for k in ('valid40', 'heldout40', 'pooled80')) + ' | BAR ' + s['bar']['verdict'])")." >> $TL
echo "$(date +%T) HELDOUT40 S27r DONE"
