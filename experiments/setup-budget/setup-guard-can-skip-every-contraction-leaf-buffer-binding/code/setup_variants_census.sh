#!/usr/bin/env bash
# TAG fable-porter (PT-65b, lead 10:0x questions (1)-(2)): (A) bit-identity, in process with setup(): S27r vs S27r_ksfirst and vs S27r_trim372
# (parity_layers_setup_v2.py, nets 35,100: predictions must be identical; flops show the trim's lazy-bind cost); (B) concurrent census pairs under the
# same load (run_census_porter_v3.sh, setup-calling client, nets 35,100): S27 vs each of S27r / S27r_ksfirst / S27r_trim372; per arm: setup wall, the
# setup log per phase (lvfuse fused, ksplit leaf buffers, bound), requests, flops (bill) and residual per net. Report stack/setup_variants_report.txt.
set -u
H=/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fable_porter_v1; W=$H/stack; P2=$(dirname $H)
NPY=/Users/matt/arc-white-box-estimation/.venv/bin/python; S27F=$P2/fable_fire_v1/fire_candS27/candS27_S26_capA_mem.py
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
bash $H/machine_gate.sh 8 setup-variants-parity
for v in ksfirst trim372; do echo "== (A) S27r vs S27r_$v"; nice -n 10 $NPY $H/parity_layers_setup_v2.py $W/stack_S27r_mem.py $W/stack_S27r_${v}_mem.py 35,100 2>&1 | grep -v "Warning\|^  \|configure(" | grep "^r\|setup "; done
for v in S27r S27r_ksfirst S27r_trim372; do
  bash $H/machine_gate.sh 9 setup-variants-$v
  echo "== (B) pair S27 vs $v at load $(sysctl -n vm.loadavg)"
  bash $H/run_census_porter_v3.sh $S27F 35,100 sv_ref_$v > $W/sv_ref_$v.out 2>&1 & a=$!
  sleep 1
  bash $H/run_census_porter_v3.sh $W/stack_${v}_mem.py 35,100 sv_$v > $W/sv_$v.out 2>&1 & b=$!
  wait $a $b
  python3 - $H $W $v <<'PY'
import json, sys
H, W, v = sys.argv[1:4]
def arm(tag):
    s = json.loads([l for l in open(f'{W}/{tag}.out') if 'setup_wall_s' in l][0].split(': ', 1)[1]); rows = {r['rank']: r for r in json.load(open(f'{H}/{tag}_census.json'))}
    return s, rows
for lab, tag in (('S27', f'sv_ref_{v}'), (v, f'sv_{v}')):
    s, rows = arm(tag)
    print(f"  {lab:14s} setup {s['setup_wall_s']:.3f} s log {[list(e) for e in s['setup_log']]} | " + ' | '.join(f"r{n}: flops {rows[n]['flops']:,} req {rows[n]['requests']:,} resid {rows[n]['residual_s']:.3f}" for n in (35, 100)))
(sa, ra), (sb, rb) = arm(f'sv_ref_{v}'), arm(f'sv_{v}')
print(f"  {v} vs S27 (same load): setup {sb['setup_wall_s'] - sa['setup_wall_s']:+.3f} s; bill r35 {100 * (rb[35]['flops'] / ra[35]['flops'] - 1):+.3f}% r100 {100 * (rb[100]['flops'] / ra[100]['flops'] - 1):+.3f}%")
PY
done
echo "SETUP VARIANTS DONE"
