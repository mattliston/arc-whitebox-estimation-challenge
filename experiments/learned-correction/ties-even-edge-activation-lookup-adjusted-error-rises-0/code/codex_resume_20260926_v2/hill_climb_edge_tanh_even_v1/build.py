"""One prospective ties-even paid lookup; held until plane parent retention."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, shlex
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent;R=P.parents[3]
BASE=H/'hill_climb_exact_plane_reuse_v1/candidate.py'
T=H/'hill_climb_combo_bd_tanh_r5_v1/native_screen.json'
ENGINE=H/'hill_climb_bd_delta_uvp_read_cache_v1/screen.py'
FS=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/lib/python3.11/site-packages/flopscope'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert pin(BASE)['sha256']=='cf36eb1f82672a4d1e4396393922f88edc434b3cc07e82bf7ec1ac8b4da6db76'
assert pin(ENGINE)['sha256']=='116793c99086668ca2c58f6288fc2f0b05d3970cc221191aae550577ceb39d5b'
s=BASE.read_text();candidate=s;edits=[]
def change(name,transform,predicate=lambda n:True):
 global candidate
 nodes=[n for n in ast.parse(candidate).body if isinstance(n,ast.FunctionDef) and n.name==name and predicate(n)];assert len(nodes)==1
 before=ast.get_source_segment(candidate,nodes[0]);after=transform(before)
 assert candidate.count(before)==1 and after!=before
 candidate=candidate.replace(before,after,1);edits.append(dict(before=before,after=after))
def edge(f):
 a='    _e2_hold.append(fnp.tanh(h,out=h))';assert f.count(a)==1
 return f.replace(a,'    _e2_hold.append(_edge_tanh_lookup(h))',1)
change('_g5_message_edge_base',edge)
def lifecycle(f):
 a='    try:\n        global _SCRATCH';assert f.count(a)==1
 f=f.replace(a,'    try:\n        _EDGE_LOOKUP_STATE.clear()\n        global _SCRATCH',1)
 a='            _E2_LAYER.pop("post_cov32", None)';assert f.count(a)==1
 return f.replace(a,'            _EDGE_LOOKUP_STATE.clear()\n'+a,1)
change('_predict_binary',lifecycle,lambda n:any(a.arg=='covariance_hook' for a in n.args.args))
suffix='\n'+(D/'fragment.py').read_text();candidate+=suffix
inverse=candidate[:-len(suffix)]
for edit in reversed(edits):assert inverse.count(edit['after'])==1;inverse=inverse.replace(edit['after'],edit['before'],1)
assert inverse==s;compile(candidate,str(D/'candidate.py'),'exec')
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),edits=edits,suffix=suffix,fragment=pin(D/'fragment.py'),exact_inverse=True))
e=ENGINE.read_text();before="(640000000 if role == 'candidate' else 512000000)"
after="(640000000 if role == 'candidate' or arm['source']['sha256'] == 'cf36eb1f82672a4d1e4396393922f88edc434b3cc07e82bf7ec1ac8b4da6db76' else 512000000)"
assert e.count(before)==1;new=e.replace(before,after,1);assert new.replace(after,before,1)==e
compile(new,str(D/'screen.py'),'exec')
with (D/'screen.py').open('x') as f:f.write(new)
once(D/'engine_inverse.json',dict(parent=pin(ENGINE),engine=pin(D/'screen.py'),before=before,after=after,exact_inverse=True))
with (D/'engine_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(e.splitlines(True),new.splitlines(True),fromfile=str(ENGINE),tofile=str(D/'screen.py'))))
retained=BASE.parent/'retained_component.json'
protocol=dict(schema='edge-tanh-even-paid-lookup-v1',root_selected=True,parent=pin(BASE),scope='native2_only_conditional_on_parent_retention',
 parent_retention_requirement=str(retained),parent_retention=pin(retained) if retained.exists() else None,
 intervention='Only active _g5_message_edge_base tanh(h,out=h), existing L8–14 calls, is replaced by one fixed uniform nearest-grid table on[0,8], step2^-20 and original sign. Other activations and all learned/physical parameters remain unchanged.',
 adaptive_selection='Root selects one new fixed ties-even variant after source proof of half-up uniform FP32 grid magnitude bias. This explicitly supersedes the old self-imposed no-follow-on plan while preserving the failed half-up trial. It does not assert that tie bias caused the observed raw regression. Same grid/clamp/L8–14, no spacing/window/scale/coefficient sweep or fit.',
 prior_failure=pin(H/'hill_climb_edge_tanh_lookup_v1/closeout.json'),prior_protocol=pin(H/'hill_climb_edge_tanh_lookup_v1/protocol.json'),tie_bias_analysis=pin(H/'edge_lookup_tie_bias_source_note_v1.json'),root_selection='Prospective parent instruction: one ties-even trial, only after exact-plane parent native8 retention; numerical execution root-only.',
 table_elements=8388609,step=2**-20,clamp=8.,selected_layers=list(range(8,15)),parity_layers=list(range(8)),fragment=pin(D/'fragment.py'),formula_fixture=pin(D/'formula_fixture.py'),
 ownership='Paid FP32 integer arange, multiply and tanh create table once per prediction on first edge call. One shape only; shape change drops all old state. One FP32 work plane plus fresh int32 cast per call. take(mode=clip,out=work) reuses work; original h sign remains intact until final copysign(out=h). State cleared at active prediction try entry and innermost finally, including exceptions. No cross-network values retained.',
 numeric_scope='Finite FP32 activations. Power-of-two scaling is exact and all table integers0..2^23 are exactly representable. Public rint uses nearest ties-even before the exact int32 cast. Proven index range0..2^23 permits modeclip. Original signed zero is restored by copysign. No claim of exact predictions.',
 proof=dict(quantization_bound=2**-21+2**-45,table_rounding_allowance=2**-25,conditional_global_bound=2**-21+2**-45+2**-25,tail_bound=2.384185791015625e-7,qualification='The quantization and saturation bounds are analytic. The5.0664e-7 total bound assumes correctly rounded FP32 table tanh; installed public wrapper forwards NumPy tanh and does not certify its elementary-function error. More generally total error is <=2^-21+2^-45+maximum table tanh primitive error; native validation remains mandatory.'),
 price=dict(old_flops_per_element=16,new_flops_per_element=10,activation_elements=117440512,table_cold_flops=159383571,constant_array_flops=4,net_saved_flops=545259497,approx_s34_bill_saving_pct=100*545259497/267405124808,temporary_storage_bytes=167772164,constant_storage_bytes=16,explanation='Runtime6unit ops(abs,min,mul,rint,astype,copysign)+take4=10; fourconstants retained including unusedhalf for minimum source difference. Table arange2+multiply1+tanh16=19F per8388609 values; constants4 extra. Storage FP32scratch67108864+int32indices67108864+table33554436; no mode-raise gather output buffer. Forecast excludes tiny metadata and measures no speed or accuracy.'),
 billing_sources=[pin(FS/p) for p in ('_array_ops.py','_ndarray.py','_pointwise.py','data/default_weights.json')],
 resources='Candidate and exactplane comparator640M internal UBCcap; historicalcontrols512M. Original150s N2/300sN6/7.5GB RSS/zeroSwap/setup4.1 and ownboundary/serving gates unchanged.',
 gates='N2 incremental>=.1%gain,>=1win,maxreg<=1%,L0–7bytes. Ownboundary then N8>=.1%gain,>=6wins,CIupper<0,additional6negative. Broad only totalS34>=.8%; finalVALID40>=1% and fullserving. No learned-target fitting.',
 parent_native2=pin(BASE.parent/'native2_assessment.json') if (BASE.parent/'native2_assessment.json').exists() else None,model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),fragment=pin(D/'fragment.py'),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),formula_fixture=pin(D/'formula_fixture.py'),engine=pin(D/'screen.py'),engine_inverse=pin(D/'engine_inverse.json'),source_inverse_exact=True,changed_parent_functions=['_g5_message_edge_base','_predict_binary'],model_executed=False))
original=json.loads(T.read_text());old_name=T.parent.name
def rename(x):
 if isinstance(x,dict):return {k:rename(v) for k,v in x.items()}
 if isinstance(x,list):return [rename(v) for v in x]
 if isinstance(x,str) and x.startswith('/'):
  return str(Path(*[D.name if v==old_name else D.name+v[len(old_name):] if v.startswith(old_name+'_') else v for v in Path(x).parts]))
 if isinstance(x,str) and x.startswith(old_name+'_'):return D.name+x[len(old_name):]
 return x
c=rename(copy.deepcopy(original));c['engine']=pin(D/'screen.py');c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=list(range(8))
parent_config=BASE.parent/'native_screen_continuation.json'
if not parent_config.exists():parent_config=BASE.parent/'native_screen.json'
pconf=json.loads(parent_config.read_text());base=copy.deepcopy(pconf['arms']['candidate'])
for size in ('2','6'):
 b=base['blocks'][size];out=H/b['tag']
 ready=False
 if (out/'rows.jsonl').exists() and Path(b['guard']).exists():
  g=json.loads(Path(b['guard']).read_text());ready=g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None and not g['remaining_rows']
 if ready:
  b['artifacts']=dict(manifest=pin(out/'manifest.json'),rows=pin(out/'rows.jsonl'),guard=pin(b['guard']),predictions={str(r):pin(out/f'pred_{r}.npy') for r in b['ranks']})
 else:b.pop('artifacts',None)
c['arms']['base']=copy.deepcopy(base);c['arms']['parity_parent']=copy.deepcopy(base)
c['arms']['candidate']['setup']=copy.deepcopy(base['setup']);c['arms']['candidate']['ubc_retired_required']=True
c['provenance']=[pin(p) for p in (BASE,T,ENGINE,D/'build.py',D/'build.json',D/'protocol.json',D/'fragment.py',D/'source_inverse.json',D/'source_diff.patch',D/'formula_fixture.py',D/'screen.py',D/'engine_inverse.json',D/'engine_diff.patch',parent_config)]+protocol['billing_sources']+[protocol['prior_failure'],protocol['prior_protocol'],protocol['tie_bias_analysis']]
if retained.exists():c['provenance'].append(pin(retained))
c['interpretation']=protocol['gates'];c['parent_retention_required']=str(retained);c['parent6_pinned']='artifacts' in base['blocks']['6']
assert c['prospective_scope']=='native2_only'
c['parent2_pinned']='artifacts' in base['blocks']['2']
c['prospective_authorization']=False
c['prospective_scope']='held_until_plane_retention_and_actual_controls'
for size in ('2','8'):assert c['stages'][size]['gates']==original['stages'][size]['gates']
once(D/'native_screen_draft.json',c)
s34=copy.deepcopy(c);s34['arms']['base']=json.loads((BASE.parent/'native_screen_s34_report.json').read_text())['arms']['base']
for size in ('2','8'):
 s34['stages'][size]['result']=str(D/f'native{size}_s34_report.json');s34['stages'][size]['assessment_guard']=str(D/f'native{size}_s34_report_guard.json')
s34['interpretation']='Descriptive S34 comparison only; incremental admission compares actual plane parent after retention.'
once(D/'native_screen_s34_draft.json',s34)
print(json.dumps({name:pin(D/name) for name in ('candidate.py','native_screen_draft.json','build.json','protocol.json')}))
