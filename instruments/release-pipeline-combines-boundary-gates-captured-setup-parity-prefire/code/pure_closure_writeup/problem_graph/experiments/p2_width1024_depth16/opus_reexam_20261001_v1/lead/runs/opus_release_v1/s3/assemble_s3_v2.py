"""[opus-release] S3 assembly v2 (--base): S3 = BASE (e.g. VN) + the winning join fragment appended LAST, then
 (a) compile + import check (module-level asserts, native python, no setup),
 (b) native diag on one seed with u64's diag_judge_v2 (PB_ALL, guard v3: footprint 7.5 GB, RSS backstop 10 GB, nice 10, BLAS 1):
     layers BEFORE the first changed join must be byte-identical to the BASE's own diag on the same seed (run here unless --base-diag;
     default join layer: first key of the base's _JOIN_SCHEDULE);
     reports the first differing layer, plain and noise-free (cross-half U) final-layer MSE change and FLOPs change vs VB,
 (c) unless --no-pipeline: launches release_pipeline_v4.py (sequential natives) on S3 with --science @list (+ these receipts).
Never fires. usage: assemble_s3_v2.py --base B [--base-sha P] [--base-diag JSON] --fragment F [--fragment-sha P] --tag TAG --desc '...'
       [--extra-science a,b|@file] [--join-layer J] [--seed S] [--no-pipeline]"""
import argparse, ast, hashlib, json, math, os, subprocess, sys
from pathlib import Path
import numpy as np
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
L = O/'lead/runs'; D = Path(__file__).resolve().parent; P = L/'opus_release_v1/pipeline'
PY = '/Users/matt/arc-white-box-estimation/.venv/bin/python'; NPY = str(O.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return subprocess.check_output(['/bin/date', '-u', '+%FT%TZ'], text=True).strip()
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
def env(**x):
    e = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **{k: '1' for k in THREADS})
    for k in ('PB_ALL','MODE','DRY','STATS','SETUP_ONLY','CAPTURED_SETUP_MANIFEST'): e.pop(k, None)
    e.update(x); return e

ap = argparse.ArgumentParser()
ap.add_argument('--fragment', required=True, type=Path); ap.add_argument('--fragment-sha', default='')
ap.add_argument('--tag', required=True); ap.add_argument('--desc', required=True)
ap.add_argument('--base', required=True, type=Path); ap.add_argument('--base-sha', default=''); ap.add_argument('--base-diag', type=Path)
ap.add_argument('--science', default='@' + str(D/'EVIDENCE_S3p_base_v1.txt')); ap.add_argument('--extra-science', default='')
ap.add_argument('--join-layer', type=int)
ap.add_argument('--seed', type=int, default=7103248); ap.add_argument('--no-pipeline', action='store_true')
a = ap.parse_args()
frag = a.fragment.resolve(); fsha = sha(frag)
VB = a.base.resolve(); VB_SHA = sha(VB); assert VB_SHA.startswith(a.base_sha), ('base sha mismatch', VB_SHA)
assert fsha.startswith(a.fragment_sha), ('fragment sha mismatch', fsha)
vb = VB.read_text()
J = a.join_layer if a.join_layer is not None else min(ast.literal_eval([l for l in vb.splitlines() if l.startswith('_JOIN_SCHEDULE = ')][-1].split(' = ', 1)[1]))
W = D/f'S3_{a.tag}'; W.mkdir()
s3 = W/f'candidate_S3_{fsha[:8]}_v1.py'; text = vb + '\n\n' + frag.read_text(); s3.write_text(text); compile(text, str(s3), 'exec')
h = sha(s3); print(f'{utc()} S3 built {h[:12]} (base {VB_SHA[:12]} + fragment {fsha[:12]})', flush=True)
# (a) import check
chk = ("import sys,types\nstub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub\n"
       "m=types.ModuleType('c');m.__file__=sys.argv[1];sys.modules['c']=m\nexec(compile(open(sys.argv[1]).read(),sys.argv[1],'exec',dont_inherit=True),m.__dict__)\n"
       "d=m.__dict__;print('IMPORT_OK',len(d['_PB_SINGLE']),all(d.get(n) is v for n,v in d['_ASTRA_DENSE_CONTEXT']),d['Estimator'].predict.__name__)\n")
r = subprocess.run(['nice', '-n', '10', NPY, '-B', '-c', chk, str(s3)], env=env(), capture_output=True, text=True, timeout=300)
imp = [l for l in r.stdout.splitlines() if l.startswith('IMPORT_OK')]
print(f'{utc()} import check rc={r.returncode} {imp}', flush=True); assert r.returncode == 0 and imp, r.stderr[-2000:]
# (b) native diag byte check vs VB on the seed
free = int(subprocess.check_output(['/usr/bin/memory_pressure'], text=True).split('free percentage: ')[1].split('%')[0]); assert free >= 60, f'memory free {free}%'
def diag(path, digest, out, tag):
    cmd = [PY, '-B', str(O/'release/guard_footprint_v3.py'), '--record', str(W/f'guard_diag_{tag}.json'), '--seconds', '900', '--footprint-limit', '7500000000',
           '--rss-backstop', '10000000000', 'nice', '-n', '10', NPY, '-B', str(L/'opus_f384_v1/diag/diag_judge_v2.py'), str(path), str(a.seed), str(out)]
    r = subprocess.run(cmd, cwd=O/'release', env=env(PB_ALL='1', CANDIDATE_SHA256=digest), capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0 and Path(out).exists(), r.stdout[-1500:] + r.stderr[-1500:]
if a.base_diag is None:
    a.base_diag = W/f'diag_base_{a.seed}.json'; diag(VB, VB_SHA, a.base_diag, 'base'); print(f'{utc()} base diag done', flush=True)
assert json.loads(Path(a.base_diag).read_text())['sha256'] == VB_SHA, 'base diag is not of this base'
out = W/f'diag_S3_{a.seed}.json'
diag(s3, h, out, 'S3')
ref_json = Path(a.base_diag).resolve(); ref_pred = Path(str(ref_json).replace('.json', '_pred.npz'))
yb = np.load(ref_pred)[str(a.seed)]; ys = np.load(str(out).replace('.json', '_pred.npz'))[str(a.seed)]
same = [bool(yb[l].tobytes() == ys[l].tobytes()) for l in range(16)]
first_diff = next((l for l in range(16) if not same[l]), None)
jb, js = json.loads(ref_json.read_text())['rows'][0], json.loads(out.read_text())['rows'][0]
with np.load(O/'analyst/fresh'/f'truth_seed{a.seed}.npz') as z: A, B = z['mc_post_A'][-1].astype(np.float64), z['mc_post_B'][-1].astype(np.float64)
U = lambda y: float(np.mean((y[15].astype(np.float64) - A) * (y[15].astype(np.float64) - B)))
res = dict(utc=utc(), S3=dict(path=str(s3), sha256=h), VB_sha256=VB_SHA, fragment=dict(path=str(frag), sha256=fsha), seed=a.seed, join_layer_declared=J,
           layers_byte_identical=same, first_diff_layer=first_diff, prefix_ok=(first_diff is None or first_diff >= J),
           mse15_plain_log_pct=100*math.log(js['mse_final']/jb['mse_final']), mse15_U_log_pct=100*math.log(U(ys)/U(yb)),
           flops_log_pct=100*math.log(js['flops']/jb['flops']), flops_S3=js['flops'], flops_VB=jb['flops'], guard=json.loads((W/'guard_diag_S3.json').read_text()).get('error'), base_diag=str(a.base_diag))
with open(W/'S3_ASSEMBLY_CHECK_v1.json', 'x') as f: json.dump(res, f, indent=2)
print(f"{utc()} byte check: first differing layer {first_diff} (declared first join L{J}) prefix_ok={res['prefix_ok']}; one-seed MSE plain {res['mse15_plain_log_pct']:+.3f}% "
      f"U {res['mse15_U_log_pct']:+.3f}%; FLOPs {res['flops_log_pct']:+.3f}%", flush=True)
assert res['prefix_ok'], 'S3 changed layers before the first join: not a join-only fragment'
if a.no_pipeline: sys.exit(0)
# (c) pipeline with the assembly receipts appended to the evidence list
lst = lambda v: (Path(v[1:]).read_text().split() if v.startswith('@') else [x for x in v.split(',') if x])
items = lst(a.science) + lst(a.extra_science)
ev = W/'EVIDENCE_S3_v1.txt'; ev.write_text('\n'.join(items + [str(frag), str(W/'S3_ASSEMBLY_CHECK_v1.json'), str(out), str(a.base_diag)]) + '\n')
r = subprocess.run([PY, '-B', str(P/'release_pipeline_v4.py'), '--source', str(s3), '--tag', a.tag, '--desc', a.desc, '--science', '@' + str(ev)], cwd=P, env=env())
print(f'{utc()} pipeline exit {r.returncode}; see {L}/opus_release_v1/rel_{a.tag}/PIPELINE.log', flush=True)
sys.exit(r.returncode)
