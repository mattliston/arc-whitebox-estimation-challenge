"""[opus-release] SAFE judging on a serial boundary pair (metadata only): fastgate pin, official120 stage_c x2, designation judge
--first-priced (SAFE gate: first-priced P(loss) <= 10%, warm <= 42,700, setup/binding/worst), plus deltas vs the primary's boundary.
usage: judge_safe_v1.py <safe_source> <safe_boundary_dir> <primary_boundary_dir> <tag>"""
import json, math, os, shutil, subprocess, sys, hashlib
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
REL = O/'release'; RUNS = REL/'runs'; OFF = O/'lead/runs/astra_smirnov_rpe_delivery_v1/official120_v1'; PY = '/Users/matt/arc-white-box-estimation/.venv/bin/python'
src, B, BP, tag = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(), sys.argv[4]
W = Path(__file__).resolve().parent; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **{k: '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')})
ex = json.loads((B/'EXITS_v1.json').read_text()); assert ex['normal_exit'] == 0 and ex['slow_exit'] == 0, ex
def run(cmd):
    r = subprocess.run(cmd, cwd=REL, env=env, capture_output=True, text=True, timeout=600); return r
r = run([PY, '-B', str(REL/'fastgate_pin_v1.py'), '--check', str(REL/'fastgate_pin_v1.json'), str(src), str(B/'normal'), str(B/'slow')]); assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
for m, flag in (('n', ['--short-setup']), ('s', ['--setup-slowdown', '1.3'])):
    dst = RUNS/f'fg_{tag}_official120_{m}_src/C0'; dst.parent.mkdir(parents=True); shutil.copytree(B/('normal' if m == 'n' else 'slow'), dst)
    out = f'c_{tag}_official120' if m == 'n' else f'{tag}_official120_slow'
    r = run([PY, '-B', str(OFF/'stage_c_official120_v1.py'), str(src), out, '--from-run', str(dst.parent)] + flag + ['--legs', 'C', '--parent-from', str(RUNS/'c_S42_bank_legs_heavy')])
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
r = run([PY, '-B', str(OFF/'designation_judge_official120_v1.py'), f'{tag}_official120', '--pair', str(RUNS/f'c_{tag}_official120'), '--slow', str(RUNS/f'{tag}_official120_slow'), '--worker', str(RUNS/'none'), '--first-priced'])
lines = [l for l in r.stdout.splitlines() if l.startswith(('  ok', '  FAIL'))]
(W/f'JUDGE_SAFE_{tag}.txt').write_text(r.stdout)
rows = {}
for name, d in (('safe', B), ('primary', BP)):
    for m in ('normal', 'slow'):
        rows[name, m] = [json.loads(x) for x in (d/m/'rows.jsonl').read_text().splitlines()]
delta = {m: dict(bill_first_pct=100*math.log(rows['safe', m][0]['flops']/rows['primary', m][0]['flops']), bill_warm_pct=100*math.log(rows['safe', m][1]['flops']/rows['primary', m][1]['flops']),
                 requests_first=rows['safe', m][0]['requests']-rows['primary', m][0]['requests'], requests_warm=rows['safe', m][1]['requests']-rows['primary', m][1]['requests'],
                 safe_first=rows['safe', m][0]['requests'], safe_warm=rows['safe', m][1]['requests']) for m in ('normal', 'slow')}
g = {m: json.loads((B/m/'guard.json').read_text()) for m in ('normal', 'slow')}
res = dict(tag=tag, safe_sha256=sha(src), judge_lines=lines, any_fail=any(l.startswith('  FAIL') for l in lines), warm_target=42700, delta_vs_primary=delta,
           guards={m: {k: g[m].get(k) for k in ('exit_code','error','max_rss_bytes','max_footprint_bytes')} for m in g})
with open(W/f'SAFE_JUDGE_RESULT_{tag}.json', 'x') as f: json.dump(res, f, indent=2)
print('\n'.join(lines)); print(json.dumps(delta, indent=1)); print('guards', res['guards'])
