import ast
from common import *
assert sha(PARENT)==PARENT_SHA
assert sha(D/'continuation_checkpoint_v81.json')=='f118087458e3816eefd82d195ea3c8b5156afa9698fb8d707275d5d2559490db'
previous=read(D/'continuation_checkpoint_v81.json');verify(previous['study']);verify(previous['official_bank'])
study=read(FIT/'closeout.json');assert study['completed'] and study['negative']
for p in study['artifacts']+study['external_pins']:verify(p)
old=D/'full_lower_covariance_edge_native_v1';copies=[]
for name in ('unit.py','reference.py'):
    source=(old/name).read_text();changed=source;edits=[]
    if name=='unit.py':
        edits=[('import importlib.util,math,sys,types','import importlib.util,math,sys,types,resource'),
               ('seal_check,guarded,relative','seal_check,done,relative'),("guarded('build')","done('build')"),
               ('zero_head_exact=True,truth_accessed=False)','zero_head_exact=True,truth_accessed=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)')]
    for a,b in edits:assert changed.count(a)==1;changed=changed.replace(a,b,1)
    restored=changed
    for a,b in reversed(edits):assert restored.count(b)==1;restored=restored.replace(b,a,1)
    assert restored==source;ast.parse(changed)
    with (HERE/name).open('x') as f:f.write(changed)
    copies.append(dict(old=pin(old/name),new=pin(HERE/name),edits=edits,whole_source_inverse=True))
triple_path=D/'full_lower_multilayer_mean_observer_v1/parent_triple/result.json';triple=read(triple_path)
assert [r['rank'] for r in triple['records']]==[35,100,35]
baselines=read(D/'covariance_head_spsa_native_v1/baselines_sealed.json')['rows'];parent={};weights={};truths={};external=[]
for rank in (35,100,36,38):
    wp=P2/f'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/{rank:04d}/weights.npy'
    weights[str(rank)]=pin(wp);external.append(wp)
    if rank in (35,100):
        r=next(x for x in triple['records'] if x['rank']==rank)
        assert sha(wp)==r['weights_sha256'] and sha(r['output'])==r['output_sha256']
    else:
        b=next(x for x in baselines if x['rank']==rank)
        for k in ('output','receipt','source','truth'):verify(b[k])
        assert b['source']==pin(PARENT) and sha(wp)==b['weights_sha256']
        r=read(b['receipt']['path'])['records'][0]
        assert r['includes_initialization'] and r['flops']==b['flops'] and r['output_sha256']==b['output']['sha256']
        parent[str(rank)]=dict(rank=rank,prediction=b['output'],receipt=b['receipt'],source=b['source'],flops=r['flops'],
            multiplier=max(.1,r['flops']/2**41),includes_initialization=True,weights_sha256=b['weights_sha256'])
        truths[str(rank)]=b['truth'];external.extend(Path(b[k]['path']) for k in ('output','receipt','source','truth'))
for r in triple['records']:
    assert sha(r['output'])==r['output_sha256'];external.append(Path(r['output']))
assert [parent[str(r)]['flops'] for r in (36,38)]==[420159401355,414511808707]
write(HERE/'inputs.json',dict(bank_triple=pin(triple_path),weights=weights,parent=parent,truths=truths,arrays_not_decoded=True))
write(HERE/'source_copies.json',dict(copies=copies,previous_checkpoint=pin(D/'continuation_checkpoint_v81.json'),
    fit_closeout=pin(FIT/'closeout.json'),old_proxy_gate_still_failed=True,stopping_decision_explicitly_revised=True))
external += [PARENT,D/'run_guarded.py',D/'official_best_331684.json',D/'continuation_checkpoint_v81.json',triple_path,
    D/'covariance_head_spsa_native_v1/baselines_sealed.json',FIT/'closeout.json',FIT/'result.json',FIT/'summary.json',
    FIT/'nonlinear_fit.json',FIT/'nonlinear_parameters.npz',FIT/'bank_parameters.npz',FIT/'models_sealed.json',
    old/'PROTOCOL.md',old/'accuracy_closeout.json',old/'unit.py',old/'reference.py',
    D/'full_lower_covariance_edge_packed_v1/native_qualified.json']
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in dict.fromkeys(external)],
    before_new_native_calls=True,before_new_target_reads=True,original_proxy_gate_preserved=True))
print(dict(sealed=True,source_copies=2,arrays_not_decoded=True),flush=True)
