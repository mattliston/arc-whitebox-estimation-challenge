"""Post-hook FP32 covariance ownership reuse plus overwritten star diagonal elision."""
from pathlib import Path
import ast,copy,difflib,hashlib,json,shlex
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent;R=P.parents[3]
BASE=H/'hill_climb_bd_delta_read_cache_cast_v1/candidate.py';PARENT=BASE
T=BASE.parent/'native_screen.json';ENGINE=H/'hill_climb_bd_delta_uvp_read_cache_v1/screen.py'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert pin(BASE)['sha256']=='d255a7bc32cf72073c3e944b21730f86b25226324341fbab35df8abc82716c3b'
assert pin(T)['sha256']=='123d74598d1a89e859465aef5e2be49c7e05cd1af488fe070f5c476fed211f7c'
assert pin(ENGINE)['sha256']=='116793c99086668ca2c58f6288fc2f0b05d3970cc221191aae550577ceb39d5b'
s=BASE.read_text();candidate=s;edits=[]
def change_function(name, transform, index=-1):
 global candidate
 tree=ast.parse(candidate);lines=candidate.splitlines(True)
 f=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name][index]
 before=''.join(lines[f.lineno-1:f.end_lineno]);after=transform(before)
 assert before!=after and candidate.count(before)==1
 candidate=candidate.replace(before,after,1);edits.append(dict(before=before,after=after))
def loop(f):
 old='                C = C.astype(fnp.float64)\n                fnp.fill_diagonal(C, _ps_Cdiag)\n'
 new='                _ps_post_cov32 = C\n                C = C.astype(fnp.float64)\n                fnp.fill_diagonal(C, _ps_Cdiag)\n                if _FULL_LOWER_MOMENTS:\n                    fnp.fill_diagonal(_ps_post_cov32, _ps_Cdiag)\n                    _E2_LAYER["post_cov32"] = (C, _ps_post_cov32)\n'
 assert f.count(old)==1;f=f.replace(old,new,1)
 old='            star = 3 * rest.T - 2 * fnp.diag(fnp.diag(rest))\n'
 assert f.count(old)==1;f=f.replace(old,'            star = 3 * rest.T\n',1)
 old='            _W5_PORT_SCOPE = None\n';assert f.count(old)==1
 return f.replace(old,'            _E2_LAYER.pop("post_cov32", None)\n'+old,1)
# This is the unique active explicit-signature loop, before its existing wrappers.
functions=[x for x in ast.parse(candidate).body if isinstance(x,ast.FunctionDef) and x.name=='_predict_binary']
index=next(i for i,x in enumerate(functions) if any(a.arg=='covariance_hook' for a in x.args.args))
change_function('_predict_binary',loop,index)
def ps32(f):
 return """def _ps32(x):
    if str(getattr(x, 'dtype', '')) == 'float32':
        return x
    owned = _E2_LAYER.get('post_cov32')
    if owned is not None and owned[0] is x:
        return owned[1]
    return fnp.asarray(x, dtype=fnp.float32)
"""
change_function('_ps32',ps32)
change_function('_flms_cov_star',lambda f:f.replace('    C32 = C.astype(fnp.float32)\n','    C32 = _ps32(C)\n',1))
change_function('_flms_covariance_diagonal',lambda f:f.replace('C.astype(fnp.float32)','_ps32(C)',1))
inverse=candidate
for edit in reversed(edits):assert inverse.count(edit['after'])==1;inverse=inverse.replace(edit['after'],edit['before'],1)
assert inverse==s;compile(candidate,str(D/'candidate.py'),'exec');ast.parse(candidate)
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),edits=edits,exact_inverse=True))
e=ENGINE.read_text();before="(640000000 if role == 'candidate' else 512000000)"
after="(640000000 if role == 'candidate' or arm['source']['sha256'] == 'd255a7bc32cf72073c3e944b21730f86b25226324341fbab35df8abc82716c3b' else 512000000)"
assert e.count(before)==1;modified=e.replace(before,after,1);assert modified.replace(after,before,1)==e
compile(modified,str(D/'screen.py'),'exec')
with (D/'screen.py').open('x') as f:f.write(modified)
with (D/'engine_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(e.splitlines(True),modified.splitlines(True),fromfile=str(ENGINE),tofile=str(D/'screen.py'))))
once(D/'engine_inverse.json',dict(parent=pin(ENGINE),engine=pin(D/'screen.py'),before=before,after=after,exact_inverse=True))
n=1024;cov_saving=15*(4*n*n-2*n);star_saving=15*(n+n*(n+1)//2+n*n)
retained=BASE.parent/'retained_component.json'
protocol=dict(schema='cov-cast-star-reuse-v1',root_selected=True,parent=pin(BASE),scope='native2_only_conditional_on_parent_retention',
 parent_retention_requirement=str(retained),parent_retention=pin(retained) if retained.exists() else None,
 intervention='Retain the final FP32 post-hook covariance after GN_RC and before widening. After existing C64 diagonal refill, patch the retained32 diagonal from the identical authoritative _ps_Cdiag, then store the exact(C64,C32) pair. _ps32 reuses only on object identity. trace_post and next full covariance or terminal diagonal reuse it. Remove only star diagonal subtraction immediately overwritten by the original fill_diagonal.',
 ownership='Full-lower lineage only. C is not mutated after the pair is bound; _t31_commit=None. Next covariance executes before _E2_LAYER.clear. Pair is replaced each layer and popped in active prediction finally. No value hashing or shape-based ancestry, no cross-network retained pair.',
 covariance_proof='Offdiagonal C64 entries are exact widenings of retained FP32 values, so narrowing returns the same bits. Diagonal C64 refill and retained32 refill use the same FP64 source values, yielding identical FP32 conversion. Preserve both original FP64 diagonals and all contraction/read ordering.',
 star_proof='rest is FP32. Parent offdiagonal subtraction is x−(+0), preserving finite values and signed zeros; parent diagonal is immediately replaced by _ps_star_diag. Child retains the original3*rest.T operation and the exact same final diagonal fill.',
 price=dict(covariance_removed_casts=30,conservative_cov_saved_flops=cov_saving,star_saved_flops=star_saving,total_saved_flops=cov_saving+star_saving,approx_s34_bill_saved_pct=100*(cov_saving+star_saving)/267405124808,explanation='30denseFP64→32 casts at rate2; charge conservative2*n extra diagonal refill per15calls (actual public destination rate is1). Star removal bills diag construction n, scalar multiply n(n+1)/2 because it is structurally symmetric, and dense subtract n² per15calls. Source forecast, not measured gain.'),
 resources='Original640M cache,5slots,7.5GB processRSS,zeroaddedSwap/setup4.1/serving limits unchanged. Assessor640M applies only candidate and exact pinned cachecast parent; older/S34 controls retain512M.',
 gates='N2 vs actual cachecast parent gain>=.02%,2wins,maxreg0,ALL16 bytes exact,allfinite/resources. Own boundary before N8 gain>=.02%,8wins,CIupper<0,additional6negative,ALL16 exact. Parent retention required before numerical acquisition. Broad totalS34>=.8%; finalVALID40 bank>=1% plus serving.',
 parity_layers=list(range(16)),parent_native2=pin(BASE.parent/'native2_assessment.json'),no_parameter_or_window_search=True,model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),engine=pin(D/'screen.py'),engine_inverse=pin(D/'engine_inverse.json'),source_inverse_exact=True,model_executed=False))
original=json.loads(T.read_text());old_name=T.parent.name
def rename(x):
 if isinstance(x,dict):return {k:rename(v) for k,v in x.items()}
 if isinstance(x,list):return [rename(v) for v in x]
 if isinstance(x,str) and x.startswith('/'):
  parts=[D.name if v==old_name else D.name+v[len(old_name):] if v.startswith(old_name+'_') else v for v in Path(x).parts];return str(Path(*parts))
 if isinstance(x,str) and x.startswith(old_name+'_'):return D.name+x[len(old_name):]
 return x
c=rename(copy.deepcopy(original));c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=list(range(16));c['engine']=pin(D/'screen.py')
bdc=json.loads((BASE.parent/'native_screen_continuation.json').read_text());base=copy.deepcopy(bdc['arms']['candidate'])
for size in ('2','6'):
 block=base['blocks'][size];folder=H/block['tag']
 if size=='2' or (folder/'rows.jsonl').exists() and Path(block['guard']).exists():
  block['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(block['guard']),predictions={str(r):pin(folder/f'pred_{r}.npy') for r in block['ranks']})
 else:
  block.pop('artifacts',None)
c['arms']['base']=copy.deepcopy(base);c['arms']['parity_parent']=copy.deepcopy(base)
c['stages']['2']['gates'].update(adjusted_pct_at_most=-.02,wins_min=2,max_per_network_regression_pct=0.0)
c['stages']['8']['gates'].update(adjusted_pct_at_most=-.02,wins_min=8,max_per_network_regression_pct=0.0)
c['provenance']=[pin(p) for p in (BASE,T,ENGINE,D/'build.py',D/'build.json',D/'protocol.json',D/'source_inverse.json',D/'source_diff.patch',D/'screen.py',D/'engine_inverse.json',D/'engine_diff.patch',BASE.parent/'native_screen_continuation.json',BASE.parent/'native2_assessment.json')]
if retained.exists():c['provenance'].append(pin(retained))
c['interpretation']=protocol['gates']
c['parent_retention_required']=str(retained)
c['parent6_pinned']='artifacts' in base['blocks']['6']
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
