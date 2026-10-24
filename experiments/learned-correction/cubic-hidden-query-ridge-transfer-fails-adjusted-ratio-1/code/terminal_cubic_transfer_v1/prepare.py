import ast
from common import *
OLD=D/'terminal_cubic_hidden_queries_v1';OBS=D/'full_lower_multilayer_mean_observer_v1'
assert sha(SOURCE)==PARENT_SHA
assert sha(D/'continuation_checkpoint_v78.json')=='44880891138b75b948f6acb087030da3979e9518e03b3880c653839082cc3822'
for k in ['study','official_bank']:check(read(D/'continuation_checkpoint_v78.json')[k])
for name in ['helper.py','unit.py']:assert (HERE/name).read_bytes()==(OLD/name).read_bytes()
source=(OLD/'capture.py').read_text();tree=ast.parse(source)
node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observed'][0]
with (HERE/'observer_source.py').open('x') as f:
    f.write('import ast,hashlib\nfrom common import *\n'+ast.get_source_segment(source,node)+'\n')
old_inputs=read(OBS/'inputs.json');train=old_inputs['train']
assert train==[36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58,60]
metadata={r['rank']:r for r in old_inputs['rows']}
training={r['rank']:r for r in read(OBS/'train_captures_completed.json')['rows']}
controls={35:read(OBS/'parent_triple/result.json')['records'][0],100:read(OBS/'parent_cold100/result.json')['records'][0]}
features={r['rank']:r for r in read(OLD/'features_result.json')['rows'][:2]}
rows=[];external=[SOURCE,D/'run_guarded.py',D/'continuation_checkpoint_v78.json',D/'official_best_331684.json',
    OLD/'closeout.json',OLD/'PROTOCOL.md',OLD/'RESULT.md',OLD/'capture.py',OLD/'helper.py',OLD/'unit.py',
    OLD/'unit.json',OLD/'features_result.json',OBS/'inputs.json',OBS/'train_captures_completed.json',
    OBS/'parent_triple/result.json',OBS/'parent_cold100/result.json',OBS/'RESULT.md']
for rank in [35,100]+train:
    if rank in controls:
        c=controls[rank];assert c['includes_initialization']
        pp=pin(c['output']);pr=pin(OBS/('parent_triple/result.json' if rank==35 else 'parent_cold100/result.json'));bill=c['flops']
    else:
        c=training[rank];pp=c['parent']['output'];pr=c['parent']['receipt'];bill=c['parent']['flops']
        assert c['record']['includes_initialization'] and bill==c['record']['flops']
        assert c['record']['source_sha256']==PARENT_SHA
    row=dict(rank=rank,weights=metadata[rank]['weights'],truth=metadata[rank]['truth'],
        parent_prediction=pp,parent_result=pr,parent_flops=bill)
    for key in ['weights','truth','parent_prediction','parent_result']:check(row[key])
    if rank in features:row['old_feature']=features[rank]
    rows.append(row)
write(HERE/'inputs.json',dict(train=train,qualification=[35,100,35],rows=rows,
    targets_decoded=False,old_oracle_coefficients_excluded=True))
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]
    +[pin(p) for p in external],targets_decoded=False,old_oracle_coefficients_excluded=True))
print(dict(sealed=True,train=len(train),cold_calls=23,feature_family_unchanged=True),flush=True)
