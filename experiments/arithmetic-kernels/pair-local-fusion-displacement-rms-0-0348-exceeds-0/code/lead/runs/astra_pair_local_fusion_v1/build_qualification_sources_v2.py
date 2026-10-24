"""Source-only builder. Does not import any candidate or execute a model."""
from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=O/'scout_native/astra_compute_v1_arm/candidate.py'
assert sha(parent)=='2aaf7aa0fd8c66ba63b895b2742a4c7640ee2c50835cd689ac9dce1f581108cb'
old=D/'fragment_plf_unvalidated_v1.py';assert sha(old)=='47ac94a6ba970e90d2d9b1b3849366c947bf9eca662db20f6ca84653662315f1'
text=old.read_text()
text=text.replace('_PLF_PARENT_SERIES = _psp_series','''_PLF_PARENT_SERIES = _psp_series
_PLF_STATS = dict(off_calls=0, on_calls=0, Q=0, H=0, local_flop_credit=0,
                  persistent_calls=0, gathered_calls=0)
''',1)
text=text.replace('    if not _PLF_ENABLED:\n        return',
                  "    if not _PLF_ENABLED:\n        _PLF_STATS['off_calls'] += 1\n        return",1)
needle="    Z = a['Z']"
addition="""    # Scalar shape accounting only: no tensor/view/context retained.
    q = 1
    for dim in a['Z'].shape[1:]:
        q *= int(dim)
    h = 1
    for dim in F.shape[1:]:
        h *= int(dim)
    _PLF_STATS['on_calls'] += 1
    _PLF_STATS['Q'] += q
    _PLF_STATS['H'] += h
    _PLF_STATS['local_flop_credit'] += 21*q + 50*h
    _PLF_STATS['persistent_calls' if persistent_fg else 'gathered_calls'] += 1
"""
assert text.count(needle)==1;text=text.replace(needle,addition+needle,1)
fragment=D/'fragment_plf_unvalidated_v2.py'
with fragment.open('x') as f:f.write(text)
held=D/'native_qualification_sources_v1';held.mkdir(exist_ok=False)
sources={'parent':{'path':str(parent),'sha256':sha(parent),'mode':None}}
for role,enabled in [('OFF',False),('ON',True)]:
    frag=text.replace('_PLF_ENABLED = True','_PLF_ENABLED = '+str(enabled),1)
    p=held/(role+'.py')
    with p.open('x') as f:f.write(parent.read_text()+'\n\n'+frag)
    ast.parse(p.read_text())
    sources[role]={'path':str(p),'sha256':sha(p),'mode':int(enabled)}
energy=O/'lead/runs/astra_gnn_energy_v1/native_energy_price_judge_UNVALIDATED_v1.py'
assert sha(energy)=='e600dcb336b5025250cb7a925664348574e90a2847580bf2181a164ed3aba65d'
j=energy.read_text()
j=j.replace("assert int(os.environ['ENERGY_MODE']) in (0,1,2)\nassert getattr(mod,'_ASTRA_GE_MODE',0)==int(os.environ['ENERGY_MODE'])", "assert int(os.environ['PLF_MODE']) in (0,1)\nassert getattr(mod,'_PLF_ENABLED',None) is bool(int(os.environ['PLF_MODE']))")
j=j.replace("'ENERGY_MODE'","'PLF_MODE'").replace("energy_mode=getattr(mod,'_ASTRA_GE_MODE',0)","plf_mode=int(mod._PLF_ENABLED)")
j=j.replace("    truth = loader.truth(seed, check_weights=False)['mc_post']", "    truth = loader.truth(seed, check_weights=False)['mc_post']\n    weights_before_sha256 = hashlib.sha256(w.tobytes()).hexdigest()")
j=j.replace("    mse = ((y - truth) ** 2).mean(axis=1)","    assert y.shape == (16,1024) and np.isfinite(y).all()\n    weights_after_sha256 = hashlib.sha256(w.tobytes()).hexdigest()\n    assert weights_before_sha256 == weights_after_sha256\n    mse = ((y - truth) ** 2).mean(axis=1)")
j=j.replace("row = dict(load_wall_s=", "row = dict(weights_before_sha256=weights_before_sha256, weights_after_sha256=weights_after_sha256, load_wall_s=")
judge=D/'native_plf_qualification_judge_v1.py'
with judge.open('x') as f:f.write(j)
ast.parse(j)
manifest={'status':'HELD_SOURCE_ONLY_NO_MODEL_IMPORT','parent_sha256':sha(parent),
    'fragment':{'path':str(fragment),'sha256':sha(fragment)},'candidates':sources,
    'judge':{'path':str(judge),'sha256':sha(judge)},
    'source_judge':{'path':str(energy),'sha256':sha(energy)},'fixed_order':[['OFF',7100136],['ON',7100136],['OFF',7100137],['ON',7100137]],
    'cap_per_layer_displacement_over_parent_RMSE':0.001,'scientific_launch_authorized':False}
p=D/'QUALIFICATION_SOURCE_MANIFEST_v1.json'
with p.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps({'manifest':str(p),'sha256':sha(p),'fragment':manifest['fragment'],'judge':manifest['judge'],'candidates':sources},indent=2))
