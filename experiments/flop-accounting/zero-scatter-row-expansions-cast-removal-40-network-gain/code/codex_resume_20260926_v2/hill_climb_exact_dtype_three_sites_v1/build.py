"""Fixed three-site public cast elision on retained plane parent; stdlib only."""
from pathlib import Path
import ast, copy, difflib, hashlib, json

D=Path(__file__).resolve().parent; H=D.parent
P=H/'hill_climb_exact_plane_reuse_v1'
S=H/'exact_dtype_three_sites_v1'
CACHE=H/'hill_climb_bd_delta_uvp_read_cache_v1'
NATIVE=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'

def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def once(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def text_once(p,s):
    with Path(p).open('x') as f:f.write(s)

def main():
    parent=pin(P/'candidate.py');retained=pin(P/'retained_component.json')
    assert parent['sha256']=='cf36eb1f82672a4d1e4396393922f88edc434b3cc07e82bf7ec1ac8b4da6db76'
    assert retained['sha256']=='91033f9f40ca019f1ff9445058f5b11e5ba1855732adb77adcfc0dff7d7ef442'
    rr=read(retained['path']);assert rr['status']=='RETAINED_COMPONENT' and rr['candidate']==parent and rr['all16_parent_bytes_exact']
    pr=read(rr['native8']['path']);assert pin(rr['native8']['path'])==rr['native8']
    assert pr['admitted'] and pr['passed_integrity'] and all(pr['gates'].values()) and pr['all16_parent_bytes_exact']
    ep=pin(S/'source_edits.json');e=read(ep['path']);assert e['reference_parent']==parent
    source=Path(parent['path']).read_text();candidate=source
    for edit in e['edits']:
        assert candidate.count(edit['before'])==1
        candidate=candidate.replace(edit['before'],edit['after'],1)
    inverse=candidate
    for edit in reversed(e['edits']):
        assert inverse.count(edit['after'])==1
        inverse=inverse.replace(edit['after'],edit['before'],1)
    assert inverse==source
    before_tree=ast.parse(source);after_tree=ast.parse(candidate)
    assert len(before_tree.body)==len(after_tree.body)
    changed=[]
    for a,b in zip(before_tree.body,after_tree.body):
        if ast.dump(a,include_attributes=False)!=ast.dump(b,include_attributes=False):
            assert isinstance(a,ast.FunctionDef) and isinstance(b,ast.FunctionDef) and a.name==b.name
            changed.append(a.name)
    assert sorted(changed)==sorted(['_lr_factor','_gn_dcov','_flms_covariance_diagonal'])
    compile(candidate,str(D/'candidate.py'),'exec');text_once(D/'candidate.py',candidate)
    text_once(D/'fragment.py',(S/'fragment.py').read_text())
    text_once(D/'source_diff.patch',''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(P/'candidate.py'),tofile=str(D/'candidate.py'))))
    once(D/'source_inverse.json',dict(parent=parent,edits=e['edits'],exact_inverse=True))
    base_engine=pin(CACHE/'screen.py');assert base_engine['sha256']=='116793c99086668ca2c58f6288fc2f0b05d3970cc221191aae550577ceb39d5b'
    old=Path(base_engine['path']).read_text()
    before="(640000000 if role == 'candidate' else 512000000)"
    after="(640000000 if role == 'candidate' or arm['source']['sha256'] == 'cf36eb1f82672a4d1e4396393922f88edc434b3cc07e82bf7ec1ac8b4da6db76' else 512000000)"
    assert old.count(before)==1;engine=old.replace(before,after,1);compile(engine,str(D/'screen.py'),'exec')
    text_once(D/'screen.py',engine)
    once(D/'engine_inverse.json',dict(parent=base_engine,engine=pin(D/'screen.py'),before=before,after=after,exact_inverse=True))
    protocol=copy.deepcopy(read(S/'protocol.json'))
    protocol.update(schema='exact-dtype-three-sites-v1',parent_choice='Retained plane cf36; EV failed N2 and is excluded.',parent=parent,
        parent_retention=retained,scope='native2_only',fragment=pin(D/'fragment.py'),source_edits=ep,
        source_inverse=pin(D/'source_inverse.json'),parent_native8=rr['native8'],model_executed=False,
        serving_qualified=False,submitted=False,full_candidate_built=True)
    once(D/'protocol.json',protocol)
    build=dict(candidate=pin(D/'candidate.py'),parent=parent,builder=pin(__file__),protocol=pin(D/'protocol.json'),
        fragment=pin(D/'fragment.py'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),
        engine=pin(D/'screen.py'),engine_inverse=pin(D/'engine_inverse.json'),source_inverse_exact=True,
        changed_parent_functions=changed,model_executed=False)
    once(D/'build.json',build)
    c=read(P/'native_screen.json');parent_config=read(P/'native_screen_continuation.json')
    parent_arm=copy.deepcopy(parent_config['arms']['candidate']);assert parent_arm['source']==parent
    for size in ('2','6'):
        block=parent_arm['blocks'][size];folder=H/block['tag']
        block['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(block['guard']),
            predictions={str(rank):pin(folder/f'pred_{rank}.npy') for rank in block['ranks']})
    c['arms']['base']=copy.deepcopy(parent_arm);c['arms']['parity_parent']=copy.deepcopy(parent_arm)
    arm=c['arms']['candidate'];arm['source']=build['candidate']
    for size in ('2','6'):
        block=arm['blocks'][size];block.pop('artifacts',None)
        block['tag']=D.name+'_train'+size+'_v1';block['guard']=str(D/f'native{size}_guard.json')
    c.update(engine=build['engine'],parent_retention_required=retained['path'],parent_retained=retained,parent6_pinned=True,
        prospective_authorization=True,prospective_scope='native2_only',interpretation=protocol['gates'])
    for size in ('2','8'):
        c['stages'][size]['result']=str(D/f'native{size}_assessment.json')
        c['stages'][size]['assessment_guard']=str(D/f'native{size}_assess_guard.json')
    req=c['boundary_before8'];req['receipt']=str(D/'boundary_not_admitted.json');req['assessment_guard']=str(D/'boundary_guard_not_admitted.json')
    req['assessment_command'][-2]=str(D/'boundary_config_not_admitted.json')
    c['provenance']=[pin(__file__),pin(D/'build.json'),pin(D/'protocol.json'),ep,pin(S/'protocol.json'),
        pin(D/'source_inverse.json'),pin(D/'engine_inverse.json'),parent,retained,rr['native8'],rr['boundary'],
        pin(P/'native_screen_continuation.json'),pin(P/'native8_assess_guard.json'),base_engine,pin(CACHE/'build.json')]
    once(D/'native_screen.json',c);cp=pin(D/'native_screen.json');b2=arm['blocks']['2']
    once(D/'commands.json',dict(config=cp,commands={
        'native2':[str(NATIVE),'-B',b2['driver']['path'],build['candidate']['path'],','.join(map(str,b2['ranks'])),b2['tag']],
        'native2_assessment':[str(NATIVE),'-B',build['engine']['path'],cp['path'],cp['sha256'],'2']},
        native2_guard=b2['guard'],native2_limits=b2['limits'],native2_assessment_guard=c['stages']['2']['assessment_guard'],
        native2_assessment_limits=c['stages']['2']['assessment_limits'],scope='N2 only; own boundary and continuation remain required'))
    print(json.dumps({k:pin(D/k) for k in ('candidate.py','native_screen.json','build.json','protocol.json','commands.json')}))

if __name__=='__main__':main()
