from pathlib import Path
import hashlib, json, os, re, statistics, subprocess
D = Path(__file__).resolve().parent
R = D.parent
repo = Path('/Users/matt/arc-white-box-estimation')
py = R.parents[2] / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
assert py.exists(), py
p = json.loads((D/'PLAN8_v1.json').read_text())
c = Path(p['candidate'])
assert hashlib.sha256(c.read_bytes()).hexdigest() == p['sha']
env = dict(os.environ)
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    env[k] = '1'
rows = []
for rank in p['ranks']:
    pressure = subprocess.check_output(['/usr/bin/memory_pressure'], text=True)
    assert int(re.search(r'System-wide memory free percentage: (\d+)%', pressure).group(1)) >= 60
    out, guard = D/f'res_{rank}_v1.json', D/f'guard_{rank}_v1.json'
    cmd = [str(repo/'.venv/bin/python'), str(R/'astra_crunch_physics_v1/guard_memory50_v1.py'), '--record', str(guard), '--seconds', '150', '--footprint-limit', '7500000000', '--rss-backstop', '7500000000', str(py), '-B', str(D/'judge_fullstack_v1.py'), str(c), str(rank), 'ctl', str(out)]
    with (D/f'run_{rank}_v1.log').open('x') as f:
        rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    g = json.loads(guard.read_text())
    assert rc == 0 and g['completed'] and g['whole_group_retired'], g
    a = json.loads(out.read_text()); b = json.loads(Path(p['controls'][str(rank)]).read_text())
    assert a['source_sha'] == p['sha'] and b['mode'] == 'ctl'
    assert a['w3']['calls'] == 15 and a['w3']['metric_builds'] == 1
    row = dict(rank=rank,raw_pct=100*(a['U']/b['U']-1),bill_pct=100*(a['flops']/b['flops']-1),adjusted_pct=100*(a['U']*a['flops']/(b['U']*b['flops'])-1))
    rows.append(row);print(json.dumps(row),flush=True)
mean = statistics.mean(x['adjusted_pct'] for x in rows)
res = dict(rows=rows,mean_adjusted_pct=mean,se_adjusted_pct=statistics.stdev(x['adjusted_pct'] for x in rows)/len(rows)**.5,sha=p['sha'])
(D/'ROOT_RESULT8_v1.json').open('x').write(json.dumps(res,indent=2)+'\n')
print(json.dumps(res),flush=True)
