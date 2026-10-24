"""[opus-retune] guarded native screen runner = opus_rankgb_v1/run_screen_v1.py (lead's screen_judge_opus_v1, PB_ALL, BLAS 1, nice 10,
guard v3 7.5 GB / RSS 10 GB) + a machine gate: launch only if memory free >= 60% AND heavy (>1 GB RSS) processes <= 8. MAXC concurrent."""
import hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
PY = '/Users/matt/arc-white-box-estimation/.venv/bin/python'; NPY = str(O.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python')
JUDGE = O/'lead/runs/opus_lead_screen_v1/screen_judge_opus_v1.py'
jobs = json.loads(Path(sys.argv[1]).read_text()); MAXC = int(sys.argv[2])
env0 = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PB_ALL='1', **{k: '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')})
free = lambda: int(re.search(r'free percentage: (\d+)%', subprocess.check_output(['/usr/bin/memory_pressure'], text=True)).group(1))
heavy = lambda: sum(1 for l in subprocess.check_output(['ps', '-axo', 'rss=,comm='], text=True).splitlines() if l.split() and int(l.split()[0]) > 1_000_000)
live = []
for j in jobs:
    out = Path(j['out']); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): continue
    while True:
        live = [p for p in live if p.poll() is None]
        if len(live) < MAXC:
            f, h = free(), heavy()
            if f >= 60 and h <= 8: break
        time.sleep(5)
    cand = j['cand']; e = dict(env0, CANDIDATE_SHA256=hashlib.sha256(Path(cand).read_bytes()).hexdigest())
    cmd = [PY, '-B', str(O/'release/guard_footprint_v3.py'), '--record', str(out.with_name('guard_' + out.stem.split('_', 1)[1] + '.json')), '--seconds', '900',
           '--footprint-limit', '7500000000', '--rss-backstop', '10000000000', 'nice', '-n', '10', NPY, '-B', str(JUDGE), cand, str(j['seed']), str(out)]
    live.append(subprocess.Popen(cmd, cwd=O/'release', env=e, stdout=open(str(out).replace('.json', '.log'), 'w'), stderr=subprocess.STDOUT))
    print(time.strftime('%H:%M:%S'), 'launch', out.parent.name, j['seed'], 'free', f, 'heavy', h, flush=True); time.sleep(4)
for p in live: p.wait()
print('SCREEN_DONE', flush=True)
