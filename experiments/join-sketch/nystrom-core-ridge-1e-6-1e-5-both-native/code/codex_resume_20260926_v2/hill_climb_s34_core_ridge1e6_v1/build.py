"""Fixed S34 future-core ridge. Source/config construction only."""
from pathlib import Path
import ast, hashlib, json

D = Path(__file__).resolve().parent
H = D.parent
P = H / 'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
T = H / 'hill_climb_s34_late_c0_95_v1/native_screen.json'
def pin(p):
    p = Path(p).resolve()
    b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
def once(p, x):
    with p.open('x') as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write('\n')
def rewrite(x):
    if isinstance(x, dict): return {k: rewrite(v) for k, v in x.items()}
    if isinstance(x, list): return [rewrite(v) for v in x]
    if isinstance(x, str): return x.replace(T.parent.name, D.name)
    return x
assert pin(P)['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(T)['sha256'] == '25405d726b632f51dbc674d09f69946edfe262b051a91d5d85ff85f9e9fb4a36'
old = P.read_text()
before = '            K = (K + K.T) * 0.5\n            R64 = unused_R.astype(fnp.float64)\n'
after = '            K = (K + K.T) * 0.5\n            fnp.fill_diagonal(K, fnp.diag(K) + (1e-6 / rank) * fnp.trace(K))\n            R64 = unused_R.astype(fnp.float64)\n'
assert old.count(before) == 1
child = old.replace(before, after)
assert child.count(after) == 1 and child.replace(after, before) == old
a, b = ast.parse(old).body, ast.parse(child).body
assert len(a) == len(b)
changed = [i for i, (x, y) in enumerate(zip(a, b)) if ast.dump(x) != ast.dump(y)]
assert len(changed) == 1 and a[changed[0]].name == '_shared_old384_push'
with (D / 'candidate.py').open('x') as f: f.write(child)
protocol = dict(root_selected=True, baseline=pin(P), fixed_relative_ridge=1e-6,
    operation='After symmetric K, add (1e-6/rank)*trace(K) to diagonal before existing solve.',
    affected_cores=[7,10], no_future_core_at_last_join=True,
    hypothesis='Regularize the FP32-derived future-core solve; accuracy benefit unknown.',
    previous_spectra_positive=True, no_exact_prior_trial_in_bounded_review=True,
    no_threshold_sweep=True, no_validation_targets_used=True,
    scope='native2_only', roster=[35,100], parity_layers=list(range(10)),
    source_inverse_exact=True, source_edit=dict(before=before,after=after),
    native2_gates=dict(adjusted_pct_at_most=-0.1,wins_min=1,max_per_network_regression_pct=1.0),
    native8_gates=dict(adjusted_pct_at_most=-0.1,wins_min=6,ci_upper_below=0,additional6_negative=True))
once(D/'protocol.json',protocol)
build = dict(candidate=pin(D/'candidate.py'),parent=pin(P),builder=pin(__file__),
    protocol=pin(D/'protocol.json'),source_inverse_exact=True,
    changed_top_level_functions=['_shared_old384_push'], numpy_torch_or_model_executed=False)
once(D/'build.json',build)
original=json.loads(T.read_text()); cfg=rewrite(original)
cfg['arms']['candidate']['source']=build['candidate']
cfg['parity_layers']=list(range(10))
cfg['provenance']=[pin(p) for p in (P,T,Path(__file__),D/'protocol.json',D/'build.json',H/'official_bank_S34_v1.json')]
cfg['interpretation']='One fixed relative 1e-6 future-core ridge on exact S34; no other changes. Original TRAIN2 gates, controls, runtime and resource limits; L0-9 parity. Native2 only; no boundary or broader qualification yet.'
assert cfg['prospective_scope']=='native2_only'
assert cfg['arms']['base']==original['arms']['base']
assert cfg['arms']['parity_parent']==original['arms']['parity_parent']
for s in ('2','8'): assert cfg['stages'][s]['gates']==original['stages'][s]['gates']
once(D/'native_screen.json',cfg)
print(json.dumps({k:pin(D/f) for k,f in [('candidate','candidate.py'),('config','native_screen.json'),('build','build.json'),('protocol','protocol.json')]}))
