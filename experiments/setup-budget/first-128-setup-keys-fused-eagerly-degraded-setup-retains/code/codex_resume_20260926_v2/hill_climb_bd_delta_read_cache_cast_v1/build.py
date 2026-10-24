"""Prospective exact cache plus public-ufunc cast elision; stdlib builder only."""
from pathlib import Path
import ast,copy,difflib,hashlib,json,shlex
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent;R=P.parents[3]
BASE=H/'hill_climb_combo_bd_linear_delta_v1/candidate.py'
PARENT=H/'hill_climb_bd_delta_uvp_read_cache_v1/candidate.py'
T=PARENT.parent/'native_screen.json';ENGINE=PARENT.parent/'screen.py'
PKG=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/lib/python3.11/site-packages/flopscope'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert pin(BASE)['sha256']=='6f104e9db5ca32dc5e7724d64e03fd969bfff9d6cb48ce2b55695a998eecd1bb'
assert pin(PARENT)['sha256']=='a7c3e06f139b9c9d2a6ec7e78f5db9d707f7f88d24de88920ffc34c8a6ca7a9e'
assert pin(T)['sha256']=='6a607892349f89e8188d7c745e13aadf81d5ca49a9e1640f032413dc85e87f31'
assert pin(ENGINE)['sha256']=='116793c99086668ca2c58f6288fc2f0b05d3970cc221191aae550577ceb39d5b'
s=PARENT.read_text();tree=ast.parse(s);lines=s.splitlines(True)
f=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_bd_update'][-1]
function_before=''.join(lines[f.lineno-1:f.end_lineno]);candidate=function_before;edits=[]
def change(before,after):
 global candidate
 assert candidate.count(before)==1
 candidate=candidate.replace(before,after,1);edits.append(dict(before=before,after=after))
change('    old = old32.astype(fnp.float64)\n    new = new32.astype(fnp.float64)\n    difference = new - old\n','    difference = fnp.subtract(new32, old32, dtype=fnp.float64)\n')
change('_e2_H(new + old)', '_e2_H(fnp.add(new32, old32, dtype=fnp.float64))')
change('    del new\n','')
inverse=candidate
for edit in reversed(edits):
 if edit['after']:assert inverse.count(edit['after'])==1;inverse=inverse.replace(edit['after'],edit['before'],1)
 else:
  anchor='    fnp.fill_diagonal(delta, 0.0)\n    return new32, delta\n';assert inverse.count(anchor)==1;inverse=inverse.replace(anchor,edit['before']+anchor,1)
assert inverse==function_before
edits=[dict(before=function_before,after=candidate)]
assert s.count(function_before)==1
candidate=s.replace(function_before,candidate,1)
st,ct=ast.parse(s),ast.parse(candidate);compile(candidate,str(D/'candidate.py'),'exec')
assert len(st.body)==len(ct.body)
changed=[a.name for a,b in zip(st.body,ct.body) if ast.dump(a)!=ast.dump(b)];assert changed==['_bd_update']
fun=[x for x in ct.body if isinstance(x,ast.FunctionDef) and x.name=='_bd_update'][-1]
assert not any(isinstance(x,ast.Name) and x.id in ('old','new') for x in ast.walk(fun))
assert '_UBC_LIMIT = 640000000' in candidate and "len(run['cache']) < 5" in candidate
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(PARENT),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(PARENT),edits=edits,deleted_line_anchor='    fnp.fill_diagonal(delta, 0.0)\n    return new32, delta\n',exact_inverse=True,only_changed_function=changed[0]))
copy_bound=7**5*(6144//32)*((256+320)//32);cast_saved=4*1024**2*15
assert copy_bound==58084992 and cast_saved==62914560 and cast_saved>copy_bound
protocol=dict(schema='bd-delta-read-cache-cast-v1',root_selected=True,parent=pin(PARENT),comparator=pin(BASE),scope='native2_only',
 explicit_adaptive_composition='New prospective root selection after standalone cache N2 was all16 exact/raw0 but failed2wins/maxreg0: mean-.0264110198%,rank35-.0623793%,rank100+.00788486%. Original standalone source, protocol and failed receipt remain unchanged. Compose the independently source-confirmed cast materialization saving with the unchanged cache; no hit/window/rank/threshold rescue.',
 intervention='Only active _bd_update removes old32.astype(float64) and new32.astype(float64), uses fnp.subtract(new32,old32,dtype=fnp.float64), replaces new+old with fnp.add(new32,old32,dtype=fnp.float64), and removes del new. Same FP64 binary arithmetic and all later operations remain.',
 public_dtype_semantics='Installed public _pointwise._counted_binary freezes dtype, resolves complete NumPy ufunc loop signature, bills heavier loop dtype and forwards the same dtype kwargs. Float32 operands convert exactly to float64 for these two arithmetic loops. No out-cast substitution or accounting bypass. Removed explicit astype calls each cost n*n at FP64rate2.',
 scalar_bound=dict(width_max=6144,extra_transpose_slots=2,depth=5,max_extra_copy_flops=copy_bound,eliminated_cast_flops=cast_saved,conservative_difference_flops=cast_saved-copy_bound,formula='7^5*(6144/32)*((256+320)/32) < 2casts*2FP64rate*1024^2*15calls',scope='Fixed width1024/depth16 native workload and original single immutable base generation. Additional transform hits only improve this copy comparison; forecast not a gate substitute.'),
 cache_preservation='All existing three D6 reconstruction slots, two added fullD5 transpose slots, original generation/owner rules and prediction-finally retirement are unchanged from a7c3e06f. Total5slots,640000000 internal bytes; original old-slot worstcase plus two readslots stays below640M.',
 resources=dict(internal_cache_bytes=640000000,cache_slots=5,process_rss_limit_bytes=7500000000,added_swap_bytes=0,setup_seconds_max=4.1,full_serving_limits_unchanged=True,controls_cache_bytes=512000000),
 gates='N2 vs retained BD-delta gain>=.02%,2wins,maxreg0,ALL16 exact,allfinite/resources. Own boundary then N8>=.02%gain,8wins,CIupper<0,additional6negative,ALL16 exact. Broad totalS34>=.8%; finalVALID40 bank>=1% plus unchanged serving/resources.',
 parity_layers=list(range(16)),baseline_retained=pin(BASE.parent/'retained_component.json'),standalone_failed_native2=pin(PARENT.parent/'native2_assessment.json'),standalone_protocol=pin(PARENT.parent/'protocol.json'),standalone_guard=pin(PARENT.parent/'native2_guard.json'),
 installed_sources=[pin(PKG/p) for p in ('_pointwise.py','_array_ops.py','_dtype_billing.py','data/default_weights.json')],no_scope_or_scale_search=True,model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(PARENT),comparator=pin(BASE),builder=pin(__file__),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),engine=pin(ENGINE),source_inverse_exact=True,model_executed=False))
original=json.loads(T.read_text());old_name=T.parent.name
def rename(x):
 if isinstance(x,dict):return {k:rename(v) for k,v in x.items()}
 if isinstance(x,list):return [rename(v) for v in x]
 if isinstance(x,str) and x.startswith('/'):
  parts=[D.name if v==old_name else D.name+v[len(old_name):] if v.startswith(old_name+'_') else v for v in Path(x).parts];return str(Path(*parts))
 if isinstance(x,str) and x.startswith(old_name+'_'):return D.name+x[len(old_name):]
 return x
c=rename(copy.deepcopy(original));c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=list(range(16));c['engine']=pin(ENGINE)
bdc=json.loads((BASE.parent/'native_screen_continuation.json').read_text());base=copy.deepcopy(bdc['arms']['candidate'])
for size in ('2','6'):
 block=base['blocks'][size];folder=H/block['tag'];block['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(block['guard']),predictions={str(r):pin(folder/f'pred_{r}.npy') for r in block['ranks']})
c['arms']['base']=copy.deepcopy(base);c['arms']['parity_parent']=copy.deepcopy(base)
c['stages']['2']['gates'].update(adjusted_pct_at_most=-.02,wins_min=2,max_per_network_regression_pct=0.0)
c['stages']['8']['gates'].update(adjusted_pct_at_most=-.02,wins_min=8,max_per_network_regression_pct=0.0)
c['provenance']=[pin(p) for p in (BASE,PARENT,T,ENGINE,D/'build.py',D/'build.json',D/'protocol.json',D/'source_inverse.json',D/'source_diff.patch',BASE.parent/'retained_component.json',BASE.parent/'native_screen_continuation.json',PARENT.parent/'native2_assessment.json',PARENT.parent/'protocol.json',PARENT.parent/'native2_guard.json')] + protocol['installed_sources']
c['interpretation']=protocol['gates']
c.pop('additional_root_review',None)
assert c['prospective_scope']=='native2_only'
once(D/'native_screen.json',c)
s34=copy.deepcopy(c);s34['arms']['base']=json.loads((T.parent/'native_screen_s34_report.json').read_text())['arms']['base']
for size in ('2','8'):
 s34['stages'][size]['result']=str(D/f'native{size}_s34_report.json');s34['stages'][size]['assessment_guard']=str(D/f'native{size}_s34_report_guard.json')
s34['interpretation']='Descriptive S34 report only; incremental gates use retained BD parent in native_screen.json.'
once(D/'native_screen_s34_report.json',s34)
native=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python';guard=P/'codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py';light=R/'.venv/bin/python'
def wrap(record,seconds,rss,cmd):return [str(light),'-B',str(guard),'--record',str(record),'--seconds',str(seconds),'--rss-limit',str(rss),*cmd]
b=c['arms']['candidate']['blocks']['2'];commands=dict(native2=wrap(b['guard'],b['limits']['seconds'],b['limits']['rss_bytes'],[str(native),'-B',b['driver']['path'],str(D/'candidate.py'),','.join(map(str,b['ranks'])),b['tag']]))
for name,file in [('assess_native2','native_screen.json'),('report_s34','native_screen_s34_report.json')]:
 cfg=json.loads((D/file).read_text());stage=cfg['stages']['2'];commands[name]=wrap(stage['assessment_guard'],stage['assessment_limits']['seconds'],stage['assessment_limits']['rss_bytes'],[str(native),'-B',cfg['engine']['path'],str(D/file),pin(D/file)['sha256'],'2'])
once(D/'commands.json',commands)
with (D/'commands.txt').open('x') as f:
 for name,command in commands.items():f.write('# '+name+'\n'+shlex.join(command)+'\n\n')
print(json.dumps({name:pin(D/name) for name in ('candidate.py','native_screen.json','build.json','protocol.json','commands.json')}))
