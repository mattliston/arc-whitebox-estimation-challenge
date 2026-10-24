"""Fixed packed-pair scatter pilot; source/JSON/AST only, no array execution."""
from pathlib import Path
import ast, copy, difflib, hashlib, json
D=Path(__file__).resolve().parent; H=D.parent
P=H/'hill_climb_scatter_forward_gather_v1'
NATIVE=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def once(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def text_once(p,s):
    with Path(p).open('x') as f:f.write(s)
def main():
    parent=pin(P/'candidate.py')
    assert parent['sha256']=='1056a1b79acedd1ab75bd711c4880b0507318db15448db48638e618baa7dcf4d'
    prior=read(P/'native8_assessment.json')
    assert prior['admitted'] and prior['passed_integrity'] and all(prior['gates'].values()) and prior['all16_parent_bytes_exact']
    before="""    hold(fnp.take(v['p16flat'], idx['upper'], axis=1, out=v['p16out']))
    f5 = hold(fnp.take(v['Fsrc'], idx['rows'], axis=3))[:, :, :, :, :, None]
    g5 = hold(fnp.take(v['Gsrc'], idx['cols'], axis=3))[:, :, :, :, None, :]
    t21 = hold(fnp.take(st['T21'], idx['upper']))
    t12 = hold(fnp.take(st['T21'], idx['transpose']))
    t22 = hold(fnp.take(st['T22'], idx['upper']))
    t31 = hold(fnp.take(st['T31'], idx['upper']))
    t13 = hold(fnp.take(st['T31'], idx['transpose']))"""
    after="""    inverse, transposed = _ppi_get(st, idx, n, block)
    _ppi_copy_p16(st['P16'], inverse, v['p16out'], idx['count'], block, hold)
    f5 = hold(fnp.take(v['Fsrc'], idx['rows'], axis=3))[:, :, :, :, :, None]
    g5 = hold(fnp.take(v['Gsrc'], idx['cols'], axis=3))[:, :, :, :, None, :]
    t21 = hold(_ppi_take(st['T21'], inverse, idx['count'], block))
    t12 = hold(_ppi_take(st['T21'], transposed, idx['count'], block))
    t22 = hold(_ppi_take(st['T22'], inverse, idx['count'], block))
    t31 = hold(_ppi_take(st['T31'], inverse, idx['count'], block))
    t13 = hold(_ppi_take(st['T31'], transposed, idx['count'], block))"""
    source=Path(parent['path']).read_text();assert source.count(before)==1
    fragment=(D/'fragment.py').read_text();edited=source.replace(before,after,1);candidate=edited+'\n'+fragment
    assert edited.replace(after,before,1)==source
    a,b=ast.parse(source),ast.parse(edited);changed=[]
    assert len(a.body)==len(b.body)
    for x,y in zip(a.body,b.body):
        if ast.dump(x,include_attributes=False)!=ast.dump(y,include_attributes=False):
            assert isinstance(x,ast.FunctionDef) and isinstance(y,ast.FunctionDef) and x.name==y.name
            changed.append(x.name)
    assert changed==['_psp_pair_series'];compile(candidate,str(D/'candidate.py'),'exec')
    text_once(D/'candidate.py',candidate)
    text_once(D/'source_diff.patch',''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(P/'candidate.py'),tofile=str(D/'candidate.py'))))
    once(D/'source_inverse.json',dict(parent=parent,edits=[dict(before=before,after=after)],suffix='\n'+fragment,fragment=pin(D/'fragment.py'),exact_inverse=True))
    # All actual arms now use original forward-only512M, so the original engine suffices.
    engine=pin(H/'native_screen_engine_v1/screen.py')
    text_once(D/'screen.py',Path(engine['path']).read_text())
    once(D/'engine_inverse.json',dict(parent=engine,engine=pin(D/'screen.py'),before='',after='',exact_inverse=True,unchanged=True))
    protocol=dict(schema='packed-pair-inverse-v1',root_selected=True,parent=parent,source_inverse=pin(D/'source_inverse.json'),
        proposal=pin(H/'packed_pair_flat_inverse_source_proposal_v1.json'),scope='native2_only',
        change='Only six packed pair gathers: threeP16 planes plus fiveT planes use shared paid inverse scatter; F/G gathers and ordered scalar arithmetic unchanged.',
        ownership='One fixed1024/128 strong idx/st-owned shape map per prediction; paid first-use construction, finallyclear on success/exception; no response value cache.',
        exactness='Upper locations are unique; transposed inverse handles arbitrary nonsymmetric matrices; duplicate destinations confined to discarded q. P16scratch copied before reuse. Existing output shapes/dtypes/layout and scalar reduction order preserved.',
        resources='7.5GB/zeroSwap and all original runtime gates unchanged. Owned extra6,946,816B; backend intp/flatten temporaries may add~8MiB. No absoluteRSS claim.',
        price=dict(cold_F=2818048,P16_copy_F=24772608,net_saved_F=119209984,forecast_pct=.045264734705579415,additional_paid_calls=73,additional_empty_calls=84),
        parity_layers=list(range(16)),gates='Exact N2>=.02%,2wins,maxreg0,ALL16; N8>=.02%,8wins,maxreg0,CI<0/additional6negative,ALL16.',
        future='No broad/submission until officialS35baseline is verified and a new combined candidate meets>=1% gain against it; accepted333099 is gradingpending at selection. CurrentFGrelease untouched.',
        model_executed=False,broad_qualified=False,serving_qualified=False,submitted=False)
    once(D/'protocol.json',protocol)
    build=dict(candidate=pin(D/'candidate.py'),parent=parent,builder=pin(__file__),fragment=pin(D/'fragment.py'),primitive=pin(D/'probe.py'),
        protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),engine=pin(D/'screen.py'),engine_inverse=pin(D/'engine_inverse.json'),
        source_inverse_exact=True,changed_parent_functions=changed,parent_native8=pin(P/'native8_assessment.json'),model_executed=False)
    once(D/'build.json',build)
    c=read(P/'native_screen.json');pc=read(P/'native_screen_continuation.json');pa=copy.deepcopy(pc['arms']['candidate']);assert pa['source']==parent
    for size in ('2','6'):
        block=pa['blocks'][size];folder=H/block['tag']
        block['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(block['guard']),predictions={str(r):pin(folder/f'pred_{r}.npy') for r in block['ranks']})
    c['arms']['base']=copy.deepcopy(pa);c['arms']['parity_parent']=copy.deepcopy(pa)
    ca=c['arms']['candidate'];ca['source']=build['candidate']
    for size in ('2','6'):
        z=ca['blocks'][size];z.pop('artifacts',None);z['tag']=D.name+'_train'+size+'_v1';z['guard']=str(D/f'native{size}_guard.json')
    for size in ('2','8'):
        c['stages'][size]['result']=str(D/f'native{size}_assessment.json');c['stages'][size]['assessment_guard']=str(D/f'native{size}_assess_guard.json')
    c.update(engine=build['engine'],prospective_authorization=True,prospective_scope='native2_only',parent6_pinned=True)
    for key in ('parent_retention_required','parent_retained'):c.pop(key,None)
    req=c['boundary_before8'];req['receipt']=str(D/'boundary_not_admitted.json');req['assessment_guard']=str(D/'boundary_guard_not_admitted.json');req['assessment_command'][-2]=str(D/'boundary_config_not_admitted.json')
    c['provenance']=[pin(D/'build.json'),pin(D/'protocol.json'),pin(D/'source_inverse.json'),pin(D/'engine_inverse.json'),pin(D/'probe.py'),protocol['proposal'],parent,
        pin(P/'native8_assessment.json'),pin(P/'native_screen_continuation.json'),pin(P/'native8_assess_guard.json'),pc['boundary_before8']['assessor']]
    once(D/'native_screen.json',c);cp=pin(D/'native_screen.json');z=ca['blocks']['2']
    once(D/'commands.json',dict(config=cp,commands={
        'primitive':[str(NATIVE),'-B',str(D/'probe.py'),pin(D/'build.json')['sha256'],str(D/'primitive.json')],
        'native2':[str(NATIVE),'-B',z['driver']['path'],build['candidate']['path'],','.join(map(str,z['ranks'])),z['tag']],
        'native2_assessment':[str(NATIVE),'-B',build['engine']['path'],cp['path'],cp['sha256'],'2']},
        primitive_guard=str(D/'primitive_guard.json'),primitive_limits=dict(seconds=60,rss_bytes=1000000000),native2_guard=z['guard'],native2_limits=z['limits'],
        native2_assessment_guard=c['stages']['2']['assessment_guard'],native2_assessment_limits=c['stages']['2']['assessment_limits'],scope='Root-only primitive thenN2; no boundary/N8 admission yet'))
    print(json.dumps({k:pin(D/k) for k in ('candidate.py','native_screen.json','build.json','protocol.json','commands.json')}))
if __name__=='__main__':main()
