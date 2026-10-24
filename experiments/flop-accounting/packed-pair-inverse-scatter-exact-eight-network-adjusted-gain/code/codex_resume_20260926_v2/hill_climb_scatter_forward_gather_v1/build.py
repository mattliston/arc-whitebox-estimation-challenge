"""Selected exact forward-cache memory repair plus reviewed two-site gather; stdlib only."""
from pathlib import Path
import ast,copy,difflib,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent;UG=H/'hill_climb_unique_row_gather_v1';SC=H/'hill_climb_exact_scatter_rows_v1';A=H/'hill_climb_combo_bd_linear_delta_v1';OLD=H/'hill_climb_scatter_forward_cache_v1'
NATIVE=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def text(p,s):
 with Path(p).open('x') as f:f.write(s)
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
parent=pin(UG/'candidate.py');assert parent['sha256']=='ead972324718e466d6f6b847d85ca4995d11acff423035bc5ba28458214e8380'
base=pin(SC/'candidate.py');assert base['sha256']=='0f7b93c4c45c447d76f04f428c1aeec066c5eb30027671653565b7c5fe3d93c3'
ancestor=pin(A/'candidate.py');asource=Path(ancestor['path']).read_text();source=Path(parent['path']).read_text();candidate=source
edits=[dict(before='_UBC_LIMIT = 640000000',after='_UBC_LIMIT = 512000000'),dict(before="if (len(run['cache']) < 5 and run['bytes'] + size <= _UBC_LIMIT",after="if (len(run['cache']) < 3 and run['bytes'] + size <= _UBC_LIMIT"),dict(before="inner = _ubc_transpose_product(Dv, Fv2, uv, 'vt')",after='inner = _shared_old384_mm(Dv, Fv2.T)'),dict(before="inner_u = _ubc_transpose_product(Su, Fu, uv, 'ut')",after='inner_u = _shared_old384_mm(Su, Fu.T)')]
for e in edits:assert candidate.count(e['before'])==1 and e['after'] in asource;candidate=candidate.replace(e['before'],e['after'],1)
inverse=candidate
for e in reversed(edits):assert inverse.count(e['after'])==1;inverse=inverse.replace(e['after'],e['before'],1)
assert inverse==source
ui=read(UG/'source_inverse.json');assert ui['parent']==base and inverse.endswith(ui['suffix']);inverse=inverse[:-len(ui['suffix'])]
for e in reversed(ui['edits']):assert inverse.count(e['after'])==1;inverse=inverse.replace(e['after'],e['before'],1)
assert inverse==Path(base['path']).read_text();compile(candidate,str(D/'candidate.py'),'exec');text(D/'candidate.py',candidate);cpin=pin(D/'candidate.py')
text(D/'fragment.py',(UG/'fragment.py').read_text());text(D/'probe.py',(UG/'probe.py').read_text())
save(D/'source_inverse.json',dict(parent=parent,ancestor=ancestor,edits=edits,exact_inverse=True,score_comparator=base,parent_inverse=pin(UG/'source_inverse.json'),exact_inverse_to_comparator=True))
text(D/'source_diff.patch',''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(UG/'candidate.py'),tofile=str(D/'candidate.py'))))
eparent=pin(SC/'screen.py');engine0=Path(eparent['path']).read_text();before="(640000000 if role == 'candidate' or arm['source']['sha256'] == 'cf36eb1f82672a4d1e4396393922f88edc434b3cc07e82bf7ec1ac8b4da6db76' else 512000000)";after="(640000000 if arm['source']['sha256'] == '0f7b93c4c45c447d76f04f428c1aeec066c5eb30027671653565b7c5fe3d93c3' else 512000000)"
assert engine0.count(before)==1;engine=engine0.replace(before,after,1);compile(engine,str(D/'screen.py'),'exec');text(D/'screen.py',engine);save(D/'engine_inverse.json',dict(parent=eparent,engine=pin(D/'screen.py'),before=before,after=after,exact_inverse=True))
failed=pin(SC/'serving_memory_failure_v1.json');worker=pin(SC/'serving_v2_worker_guard.json');retained=pin(SC/'retained_component.json')
protocol=dict(schema='scatter-forward-cache-plus-unique-gather-v1',root_selected=True,scope='native2_only',source_parent=parent,score_comparator=base,ancestor=ancestor,root_selection='Prospective supersession: combine reviewed UG exact cost saving with memory repair; ordinary exact gain gates. Resource-only positive-gate draft is preserved and never executed.',superseded_resource_only=dict(candidate=pin(OLD/'candidate.py'),config=pin(OLD/'native_screen.json'),build=pin(OLD/'build.json')),preserved_serving_memory_failure=failed,preserved_worker_guard=worker,changes=edits,unique_gather=pin(UG/'protocol.json'),unique_gather_review=pin(UG/'independent_source_review.json'),memory='Restore original forward-only512MB/3slots and exact original inner GEMM calls; expected~220MB removal of full-transpose copies. Current SC20th-worker RSS exceeded7.5GB; actual full20 requalification required, no leak or success presumed.',forecast='UG saved-shape~.225% less historical~.026% transpose-cache benefit gives~.199% net; forecast only, no measured new result. UG extra live storage≤131KB.',gates='Original exact native2 >=.02% adjusted gain,2wins,maxreg0,ALL16 parity vsSC. Own boundary then full8 >=.02%,8wins,maxreg0,CIupper<0,additional6negative,ALL16. Total native8 gainS34>=.8 before fixed40. Fixed40>=1% and complete20-network serving/resources unchanged.',resources=dict(candidate_cache_bytes=512000000,candidate_slots=3,exact_SC_control_cache_bytes=640000000,other_controls_cache_bytes=512000000,process_rss_bytes=7500000000,added_swap_bytes=0),no_positive_gate_exception=True,model_executed=False,submitted=False)
save(D/'protocol.json',protocol)
build=dict(candidate=cpin,parent=parent,score_comparator=base,ancestor=ancestor,builder=pin(__file__),protocol=pin(D/'protocol.json'),fragment=pin(D/'fragment.py'),primitive=pin(D/'probe.py'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),parent_build=pin(UG/'build.json'),parent_inverse=pin(UG/'source_inverse.json'),engine=pin(D/'screen.py'),engine_inverse=pin(D/'engine_inverse.json'),source_inverse_exact=True,model_executed=False)
save(D/'build.json',build)
c=read(SC/'native_screen.json');pc=read(SC/'native_screen_continuation.json');pa=copy.deepcopy(pc['arms']['candidate']);assert pa['source']==base
for size in ['2','6']:
 spec=pa['blocks'][size];folder=H/spec['tag'];g=read(spec['guard']);assert g['completed'] and g['exit_code']==0 and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
 spec['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(spec['guard']),predictions={str(rank):pin(folder/f'pred_{rank}.npy') for rank in spec['ranks']})
c['arms']['base']=copy.deepcopy(pa);c['arms']['parity_parent']=copy.deepcopy(pa);ca=c['arms']['candidate'];ca['source']=cpin
for size in ['2','6']:
 spec=ca['blocks'][size];spec.pop('artifacts',None);spec['tag']=D.name+'_train'+size+'_v1';spec['guard']=str(D/f'native{size}_guard.json')
for size in ['2','8']:
 c['stages'][size]['result']=str(D/f'native{size}_assessment.json');c['stages'][size]['assessment_guard']=str(D/f'native{size}_assess_guard.json')
c.update(engine=build['engine'],prospective_authorization=True,prospective_scope='native2_only',source_parent=parent,parent_retained=retained,parent_retention_required=retained['path'],parent2_pinned=True,parent6_pinned=True,interpretation=protocol['gates'])
c.pop('auxiliary_increment_before_continuation',None)
q=c['boundary_before8'];q['receipt']=str(D/'boundary_not_admitted.json');q['assessment_guard']=str(D/'boundary_guard_not_admitted.json');q['assessment_command'][-2]=str(D/'boundary_config_not_admitted.json')
c['provenance']=[pin(__file__),pin(D/'build.json'),pin(D/'protocol.json'),pin(D/'source_inverse.json'),pin(D/'engine_inverse.json'),parent,base,ancestor,failed,worker,retained,pin(SC/'native8_assessment.json'),pin(SC/'native_screen_continuation.json'),eparent,pin(SC/'build.json'),pin(UG/'build.json'),pin(UG/'source_inverse.json'),pin(UG/'independent_source_review.json')]
save(D/'native_screen.json',c);conf=pin(D/'native_screen.json');spec=ca['blocks']['2']
save(D/'commands.json',dict(config=conf,commands={'primitive':[str(NATIVE),'-B',str(D/'probe.py'),pin(D/'build.json')['sha256'],str(D/'primitive.json')],'native2':[str(NATIVE),'-B',spec['driver']['path'],cpin['path'],'35,100',spec['tag']],'native2_assessment':[str(NATIVE),'-B',build['engine']['path'],conf['path'],conf['sha256'],'2']},primitive_guard=str(D/'primitive_guard.json'),primitive_limits=dict(seconds=60,rss_bytes=1000000000),native2_guard=spec['guard'],native2_limits=spec['limits'],native2_assessment_guard=c['stages']['2']['assessment_guard'],native2_assessment_limits=c['stages']['2']['assessment_limits'],scope='Root-only paid primitive then N2; ownfullqualificationrequired'))
save(OLD/'superseded_source_only_v1.json',dict(status='SUPERSEDED_WITHOUT_EXECUTION',candidate=pin(OLD/'candidate.py'),config=pin(OLD/'native_screen.json'),replacement=cpin,replacement_config=conf,reason='Root prospectively selected reviewed exact UG savings plus same memory correction, preserving ordinary negative gain gates. Positive-gate engine never executed.',model_executed=False))
print(json.dumps({k:pin(D/k) for k in ['candidate.py','native_screen.json','build.json','protocol.json','commands.json']}))
