"""Prospective two-site exact gather scaffold; source-only and execution held."""
from pathlib import Path
import ast,copy,difflib,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent;P=H/'hill_climb_exact_scatter_rows_v1';CACHE=H/'hill_climb_bd_delta_uvp_read_cache_v1'
NATIVE=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def text(p,s):
 with Path(p).open('x') as f:f.write(s)
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
parent=pin(P/'candidate.py');assert parent['sha256']=='0f7b93c4c45c447d76f04f428c1aeec066c5eb30027671653565b7c5fe3d93c3'
retained=pin(P/'retained_component.json');assert retained['sha256']=='5b8b859854ce8ee3d0f6ca2f96ecad8ddc21c9c9d8ebb629e27c6f50216e8129'
ret=read(retained['path']);assert ret['status']=='RETAINED_COMPONENT' and ret['candidate']==parent and ret['all16_parent_bytes_exact']
a=read(ret['native8']['path']);assert pin(ret['native8']['path'])==ret['native8'] and a['admitted'] and a['passed_integrity'] and all(a['gates'].values())
source=Path(parent['path']).read_text();candidate=source
edits=[dict(before='left2 = _e2_pp2[response_indices]',after='left2 = _unique_rows_via_scatter(_e2_pp2, response_indices)'),dict(before="right = pp[live_indices, int(shared_old['columns']):]",after="right = _unique_rows_via_scatter(pp[:, int(shared_old['columns']):], live_indices)")]
for e in edits:assert candidate.count(e['before'])==1;candidate=candidate.replace(e['before'],e['after'],1)
suffix='\n'+(D/'fragment.py').read_text();candidate+=suffix
inverse=candidate[:-len(suffix)]
for e in reversed(edits):assert inverse.count(e['after'])==1;inverse=inverse.replace(e['after'],e['before'],1)
assert inverse==source;compile(candidate,str(D/'candidate.py'),'exec')
a0=ast.parse(source);a1=ast.parse(candidate);assert len(a1.body)==len(a0.body)+1
changed=[]
for x,y in zip(a0.body,a1.body):
 if ast.dump(x,include_attributes=False)!=ast.dump(y,include_attributes=False):assert isinstance(x,ast.FunctionDef) and x.name==y.name=='_predict_binary';changed.append(x.name)
assert changed==['_predict_binary']
text(D/'candidate.py',candidate);save(D/'source_inverse.json',dict(parent=parent,edits=edits,suffix=suffix,exact_inverse=True));text(D/'source_diff.patch',''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(P/'candidate.py'),tofile=str(D/'candidate.py'))))
base_engine=pin(CACHE/'screen.py');assert base_engine['sha256']=='116793c99086668ca2c58f6288fc2f0b05d3970cc221191aae550577ceb39d5b'
e0=Path(base_engine['path']).read_text();before="(640000000 if role == 'candidate' else 512000000)";after="(640000000 if role == 'candidate' or arm['source']['sha256'] == '0f7b93c4c45c447d76f04f428c1aeec066c5eb30027671653565b7c5fe3d93c3' else 512000000)"
assert e0.count(before)==1;engine=e0.replace(before,after,1);compile(engine,str(D/'screen.py'),'exec');text(D/'screen.py',engine);save(D/'engine_inverse.json',dict(parent=base_engine,engine=pin(D/'screen.py'),before=before,after=after,exact_inverse=True))
proposal=pin(H/'unique_row_gather_scatter_source_proposal_v1.json');protocol=dict(schema='prospective-two-site-unique-row-scatter-v1',root_selected=True,scope='HELD_SOURCE_ONLY',execution_authorized=False,activation_condition='Root may activate only after current0f7 bank qualifies/submits; no activation or model work in this scaffold.',parent=parent,parent_retention=retained,proposal=proposal,edits=edits,coefficient_or_window_search=False,proof='Both row rosters come from select argsort slices, so each source row is unique. Inverse maps requested row[j] to j; every returned destination receives exactly one write. Omitted rows overwrite only discarded sentinel k. No order assumption is needed for discarded writes. Public scatter bills all m*width writes; fresh row-major output prefix preserves layout and selected bits, with no borrowed cache grant.',cost='Old4*k*width. Newm*width + conservative2m+6k int64 metadata. Fixed shape-only arithmetic comparison selects scatter only when strictly cheaper; otherwise exact original gather. Both selected sites only; no routing/kernel/parameter/rank changes.',forecast=read(proposal['path'])['forecast'],forecast_caveat=read(proposal['path'])['forecast_caveat'],gates='Native2 >=.02% adjusted gain,2wins,maxreg0,ALL16 byte parity vs actual0f7; own boundary then full8 >=.02%,8wins,maxreg0,CIupper<0,additional6negative,ALL16 parity. Final bank >=1% full40 relative to then-current official bank and all original serving/resource gates. Current0f7 qualification and release files unchanged.',source_ownership='A is never modified. Output is a fresh allocation with one discarded width-row, held only for existing left2/right lifetimes. int64 inverse/range are call-local. No global state or cache added.',primitive='Root-only source-bound paid fixture, noncontiguous source, signed zero/subnormal, unsorted/full/empty rosters, discarded duplicate destinations and no source alias; held with execution.',model_executed=False,arrays_executed=False,submitted=False)
save(D/'protocol.json',protocol)
build=dict(candidate=pin(D/'candidate.py'),parent=parent,builder=pin(__file__),fragment=pin(D/'fragment.py'),primitive=pin(D/'probe.py'),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),engine=pin(D/'screen.py'),engine_inverse=pin(D/'engine_inverse.json'),source_inverse_exact=True,model_executed=False)
save(D/'build.json',build)
c=read(P/'native_screen.json');pc=read(P/'native_screen_continuation.json');pa=copy.deepcopy(pc['arms']['candidate']);assert pa['source']==parent
for size in ['2','6']:
 spec=pa['blocks'][size];folder=H/spec['tag'];g=read(spec['guard']);assert g['completed'] and g['exit_code']==0 and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
 spec['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(spec['guard']),predictions={str(rank):pin(folder/f'pred_{rank}.npy') for rank in spec['ranks']})
c['arms']['base']=copy.deepcopy(pa);c['arms']['parity_parent']=copy.deepcopy(pa);ca=c['arms']['candidate'];ca['source']=build['candidate']
for size in ['2','6']:
 spec=ca['blocks'][size];spec.pop('artifacts',None);spec['tag']=D.name+'_train'+size+'_v1';spec['guard']=str(D/f'native{size}_guard.json')
for size in ['2','8']:
 c['stages'][size]['result']=str(D/f'native{size}_assessment.json');c['stages'][size]['assessment_guard']=str(D/f'native{size}_assess_guard.json')
c.update(engine=build['engine'],prospective_authorization=False,prospective_scope='held_until_current_bank_qualifies_or_submits',parent_retained=retained,parent_retention_required=retained['path'],parent2_pinned=True,parent6_pinned=True,source_parent=parent,interpretation=protocol['gates'])
c.pop('auxiliary_increment_before_continuation',None)
q=c['boundary_before8'];q['receipt']=str(D/'boundary_not_admitted.json');q['assessment_guard']=str(D/'boundary_guard_not_admitted.json');q['assessment_command'][-2]=str(D/'boundary_config_not_admitted.json')
c['provenance']=[pin(__file__),pin(D/'build.json'),pin(D/'protocol.json'),pin(D/'source_inverse.json'),pin(D/'engine_inverse.json'),parent,retained,ret['native8'],ret['boundary'],pin(P/'native_screen_continuation.json'),base_engine,pin(CACHE/'build.json'),proposal]
save(D/'native_screen_draft.json',c);cp=pin(D/'native_screen_draft.json');spec=ca['blocks']['2']
save(D/'commands_draft.json',dict(status='HELD_DO_NOT_EXECUTE',scope=protocol['activation_condition'],draft_config=cp,commands={'primitive':[str(NATIVE),'-B',str(D/'probe.py'),pin(D/'build.json')['sha256'],str(D/'primitive.json')],'native2':[str(NATIVE),'-B',spec['driver']['path'],build['candidate']['path'],'35,100',spec['tag']],'native2_assessment':[str(NATIVE),'-B',build['engine']['path'],str(D/'native_screen.json'),'FUTURE_ACTIVATED_CONFIG_SHA256','2']},primitive_guard=str(D/'primitive_guard.json'),primitive_limits=dict(seconds=60,rss_bytes=1000000000),native2_guard=spec['guard'],native2_limits=spec['limits'],native2_assessment_guard=c['stages']['2']['assessment_guard'],native2_assessment_limits=c['stages']['2']['assessment_limits']))
print(json.dumps({k:pin(D/k) for k in ['candidate.py','native_screen_draft.json','build.json','protocol.json','commands_draft.json']}))
