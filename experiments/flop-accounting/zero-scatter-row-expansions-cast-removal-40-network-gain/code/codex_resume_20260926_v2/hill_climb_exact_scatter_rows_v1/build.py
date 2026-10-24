"""Selected exact row scatter + passed three-cast component; stdlib only."""
from pathlib import Path
import ast, copy, difflib, hashlib, json
D=Path(__file__).resolve().parent;H=D.parent
P=H/'hill_climb_exact_dtype_three_sites_v1';B=H/'hill_climb_exact_plane_reuse_v1'
NATIVE=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def text(p,s):
 with Path(p).open('x') as f:f.write(s)
parent=pin(P/'candidate.py');assert parent['sha256']=='279ae9e36c5a8c8875a47afb75d7e56add57973dc106e0bfb31263bc98c44e0c'
prior=read(P/'native2_assessment.json');assert prior['candidate']==parent and prior['admitted'] and prior['passed_integrity'] and prior['all16_parent_bytes_exact'] and all(prior['gates'].values())
for f in ['native2_guard.json','native2_assess_guard.json']:
 g=read(P/f);assert g['completed'] and g['exit_code']==0 and g['error'] is None and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
source=Path(parent['path']).read_text();candidate=source;edits=[]
def edit(before,after):
 global candidate
 assert candidate.count(before)==1
 candidate=candidate.replace(before,after,1);edits.append(dict(before=before,after=after))
# Preserve both captured OTT definition and final late-common override.
for node in ast.parse(source).body:
 if not isinstance(node,ast.FunctionDef):continue
 before=ast.get_source_segment(source,node);after=before
 if node.name=='_deadcut_push' and 'pp_rows' in before:
  old="    inverse = fnp.full(n, kept, dtype=fnp.int64)\n    fnp.put(inverse, rows, fnp.arange(kept, dtype=fnp.int64))\n    pp = fnp.concatenate([pp_rows, fnp.zeros((1, int(pp_rows.shape[1])), dtype=pp_rows.dtype)], axis=0)[inverse]"
  assert old in after;after=after.replace(old,"    pp = _exact_expand_rows(pp_rows, rows, n)")
  old="state['pre_basis'] = fnp.concatenate([basis, fnp.zeros((1, int(basis.shape[1])), dtype=basis.dtype)], axis=0)[inverse]"
  assert old in after;after=after.replace(old,"state['pre_basis'] = _exact_expand_rows(basis, rows, n)")
 elif node.name=='_slr_reconstruct' and 'return fnp.concatenate([old, fnp.zeros((1, columns)' in before:
  old="    inverse = fnp.full(m, kept, dtype=fnp.int64)\n    fnp.put(inverse, rows, fnp.arange(kept, dtype=fnp.int64))\n"
  assert old in after;after=after.replace(old,'')
  after=after.replace("return fnp.concatenate([old, fnp.zeros((1, columns), dtype=old.dtype)], axis=0)[inverse]","return _exact_expand_rows(old, rows, m)")
 elif node.name=='_push_bank_shared_newborn' and '_YNT_REPAIR' in before:
  old="        inverse = fnp.full(m, kept, dtype=fnp.int64)\n        fnp.put(inverse, rows, fnp.arange(kept, dtype=fnp.int64))\n        answer = fnp.concatenate((pushed, fnp.zeros((1, width), dtype=pushed.dtype)), axis=0)[inverse]"
  assert old in after;after=after.replace(old,"        answer = _exact_expand_rows(pushed, rows, m)")
 if after!=before:edit(before,after)
assert len(edits)==4
suffix='\n'+(D/'fragment.py').read_text();candidate+=suffix;compile(candidate,str(D/'candidate.py'),'exec')
inverse=candidate[:-len(suffix)]
for e in reversed(edits):assert inverse.count(e['after'])==1;inverse=inverse.replace(e['after'],e['before'],1)
assert inverse==source
text(D/'candidate.py',candidate);save(D/'source_inverse.json',dict(parent=parent,edits=edits,suffix=suffix,exact_inverse=True))
text(D/'source_diff.patch',''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(P/'candidate.py'),tofile=str(D/'candidate.py'))))
engine=(P/'screen.py').read_text();text(D/'screen.py',engine);ei=read(P/'engine_inverse.json');ei['engine']=pin(D/'screen.py');save(D/'engine_inverse.json',ei)
# Forecast uses old shape metadata; actual current census confirms call counts only.
records=read(H/'young_read_shapes_train2_v1/shape_census.json')['records'];census=read(H/'hill_climb_exact_plane_reuse_v1_boundary_feasibility_v1_census.json');forecast=[]
for rank in [35,100]:
 total=0;previous=1024;dead=basis=tails=0
 for r in [r for r in records if r['rank']==rank]:
  total+=2*previous;previous=r['response_rows']
  if r['compact']:
   dead+=(4*1024+1)*total;basis+=(4*1024+1)*(320 if 10<=r['layer']<=12 else 384)
  if r['layer']>=7:tails+=(4*r['pushed_rows']+1)*total
 bill=next(x['flops'] for x in census if x['rank']==rank)
 forecast.append(dict(rank=rank,deadcut=dead,basis=basis,old_young=tails,total_saved_flops=dead+basis+tails,percent_current_bill=100*(dead+basis+tails)/bill))
protocol=dict(schema='exact-scatter-rows-v1',root_selected=True,scope='native2_only',parent=parent,score_comparator=pin(B/'candidate.py'),parent_native2=pin(P/'native2_assessment.json'),parent_retained=pin(B/'retained_component.json'),selected_sites=['active deadcut pp and pre_basis','captured OTT old expansion','final LCR old expansion','YNT young expansion'],proof='Rows are unique argsort slices, mapped to the identical physical destination. Fresh FP32 zeros plus paid put_along_axis copies every selected row without arithmetic; omitted rows stay positive zero and selected signed zeros/subnormals are preserved. No modified rosters, reconstruction kernels, repairs, shape/layout, caches or approximation.',tariff='Old (kept+1)*width concatenate plus 4*m*width gather. New kept*width put_along_axis, free zeros and basic rows[:,None] view. Saving (4*m+1)*width; removed inverse metadata work omitted conservatively.',forecast=forecast,forecast_caveat='Historical source5d439 shape roster; current exact-plane boundary confirms 6/7 deadcut,3OTT+5LCR and8YNT calls. Widths/m may differ; forecast is not measured candidate gain. Basic metadata allowance <.0001%.',historical_shapes=pin(H/'young_read_shapes_train2_v1/shape_census.json'),historical_manifest=pin(H/'young_read_shapes_train2_v1/manifest.json'),current_census=pin(H/'hill_climb_exact_plane_reuse_v1_boundary_feasibility_v1_census.json'),prior_tariff_check=pin(H/'exact_gather_tariff_review_v1.json'),gates='Exact combined N2 >=.02%,2wins,maxreg0,all16bytes vsPR; additionally require positive bill increment and all16bytes vsactual threecastN2 before continuation. Own boundary then N8 >=.02%,8wins,maxreg0,CIupper<0,additional6negative,all16bytes vsPR. Broad>=.8% totalS34 admission; final VALID40>=1% and original serving/resources.',no_individual_threecast_native8=True,primitive='Root-only paid probe.py covers unsorted/empty/full rows,signed zero,subnormal and row-major ownership.',model_executed=False,submitted=False)
save(D/'protocol.json',protocol)
build=dict(candidate=pin(D/'candidate.py'),parent=parent,builder=pin(__file__),protocol=pin(D/'protocol.json'),fragment=pin(D/'fragment.py'),primitive=pin(D/'probe.py'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),engine=pin(D/'screen.py'),engine_inverse=pin(D/'engine_inverse.json'),source_inverse_exact=True,model_executed=False)
save(D/'build.json',build)
c=read(P/'native_screen.json');arm=c['arms']['candidate'];arm['source']=build['candidate']
for size in ['2','6']:
 arm['blocks'][size]['tag']=D.name+'_train'+size+'_v1';arm['blocks'][size]['guard']=str(D/f'native{size}_guard.json')
for size in ['2','8']:
 c['stages'][size]['result']=str(D/f'native{size}_assessment.json');c['stages'][size]['assessment_guard']=str(D/f'native{size}_assess_guard.json')
c['engine']=build['engine'];c['interpretation']=protocol['gates'];c['source_parent']=parent
q=c['boundary_before8'];q['receipt']=str(D/'boundary_not_admitted.json');q['assessment_guard']=str(D/'boundary_guard_not_admitted.json');q['assessment_command'][-2]=str(D/'boundary_config_not_admitted.json')
c['provenance']+=[pin(__file__),pin(D/'build.json'),pin(D/'protocol.json'),pin(D/'source_inverse.json'),pin(D/'engine_inverse.json'),pin(P/'native2_assessment.json'),pin(P/'native2_guard.json'),pin(P/'native2_assess_guard.json')]
c['auxiliary_increment_before_continuation']=dict(source=parent,assessment=pin(P/'native2_assessment.json'),all16_byte_parity_required=True,bill_gain_strictly_positive_required=True,result=str(D/'native2_increment_review.json'))
save(D/'native_screen.json',c);cp=pin(D/'native_screen.json');b2=arm['blocks']['2']
save(D/'commands.json',dict(config=cp,commands={'primitive':[str(NATIVE),'-B',str(D/'probe.py'),pin(D/'build.json')['sha256'],str(D/'primitive.json')],'native2':[str(NATIVE),'-B',b2['driver']['path'],build['candidate']['path'],'35,100',b2['tag']],'native2_assessment':[str(NATIVE),'-B',build['engine']['path'],cp['path'],cp['sha256'],'2']},primitive_guard=str(D/'primitive_guard.json'),primitive_limits={'seconds':60,'rss_bytes':1000000000},native2_guard=b2['guard'],native2_limits=b2['limits'],native2_assessment_guard=c['stages']['2']['assessment_guard'],native2_assessment_limits=c['stages']['2']['assessment_limits'],scope='Root-only serial primitive then native2; no continuation yet'))
print(json.dumps({k:pin(D/k) for k in ['candidate.py','native_screen.json','build.json','protocol.json','commands.json']}))
