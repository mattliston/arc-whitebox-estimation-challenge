"""Exact pre-plane retention, direct CP32 scatter and FP64 born loop into FP32 storage."""
from pathlib import Path
import ast,copy,difflib,hashlib,json,shlex
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent;R=P.parents[3]
BASE=H/'hill_climb_cov_cast_star_reuse_v1/candidate.py';PARENT=BASE
T=BASE.parent/'native_screen.json';ENGINE=H/'hill_climb_bd_delta_uvp_read_cache_v1/screen.py'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert pin(BASE)['sha256']=='4ec2b3711b3ac8519eb0fcfa990ed3110e7a4a1f8407ff5ad776a494a2302965'
assert pin(T)['sha256']=='34dd4630c8337951cbcb7384324acbaa4ad24958016f0ded5d2cd257db6f6769'
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
 def one(a,b):
  nonlocal f
  assert f.count(a)==1,a
  f=f.replace(a,b,1)
 one("            _bb_star_pre = None   # E-35: set by the batched covariance/star call\n", "            _bb_star_pre = None   # E-35: set by the batched covariance/star call\n            _plane_cov32 = _plane_star32 = None\n")
 one("                    Cpre, _bb_star_pre = _flms_cov_star(w32, C, star)   # E-35: batched with the layer's star products\n", "                    if 1 <= l <= 14:\n                        Cpre, _bb_star_pre, _plane_cov32, _plane_star32 = _flms_cov_star(w32, C, star, retain32=True)\n                    else:\n                        Cpre, _bb_star_pre = _flms_cov_star(w32, C, star)\n")
 a="            _E2_LAYER.update(mu=mu, Cpre=Cpre, vc=_e2_vc, s=_e2_s, a=_e2_a, g=response_gate)\n"
 one(a,a+"            if _plane_cov32 is not None:\n                _E2_LAYER['ps_Cpre32'] = _plane_cov32\n")
 one("            cp_pre = fnp.zeros((n, n))\n", "            cp_pre = fnp.zeros((n, n), dtype=fnp.float32)\n")
 one("                response_plane = _e2_H(_cp_read_young_sq(left2, right, lam, live_indices, shared_old)).astype(fnp.float64)\n", "                response_plane = _e2_H(_cp_read_young_sq(left2, right, lam, live_indices, shared_old))\n                if str(response_plane.dtype) != 'float32':\n                    raise TypeError('Exact CP plane retention requires float32 response')\n")
 one("                fnp.fill_diagonal(cp_pre, diagonal)\n", "                fnp.fill_diagonal(cp_pre, diagonal)\n                _plane_cpdiag64 = diagonal\n            else:\n                _plane_cpdiag64 = fnp.zeros(n, dtype=fnp.float64)\n")
 assert f.count('fnp.diag(cp_pre)')==2
 f=f.replace('fnp.diag(cp_pre)','_plane_cpdiag64',2)
 one("(_ps_cp32 + _ps32(star_pre) + fnp.outer", "(_ps_cp32 + (_plane_star32 if _plane_star32 is not None else _ps32(star_pre)) + fnp.outer")
 one("            born = born + _bd_plane_delta\n", "            born = fnp.add(born, _bd_plane_delta, dtype=fnp.float64, out=born)\n")
 return f
functions=[x for x in ast.parse(candidate).body if isinstance(x,ast.FunctionDef) and x.name=='_predict_binary']
index=next(i for i,x in enumerate(functions) if any(a.arg=='covariance_hook' for a in x.args.args))
change_function('_predict_binary',loop,index)
def covariance(f):
 a='def _flms_cov_star(w, C, A):';assert f.count(a)==1
 f=f.replace(a,'def _flms_cov_star(w, C, A, retain32=False):',1)
 a='    return Cpre, result\n';assert f.count(a)==1
 return f.replace(a,'    if retain32:\n        fnp.fill_diagonal(sub, diagonal)\n        return Cpre, result, cov, sub\n'+a,1)
change_function('_flms_cov_star',covariance)
inverse=candidate
for edit in reversed(edits):assert inverse.count(edit['after'])==1;inverse=inverse.replace(edit['after'],edit['before'],1)
assert inverse==s;compile(candidate,str(D/'candidate.py'),'exec');ast.parse(candidate)
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),edits=edits,exact_inverse=True))
e=ENGINE.read_text();before="(640000000 if role == 'candidate' else 512000000)"
after="(640000000 if role == 'candidate' or arm['source']['sha256'] == '4ec2b3711b3ac8519eb0fcfa990ed3110e7a4a1f8407ff5ad776a494a2302965' else 512000000)"
assert e.count(before)==1;modified=e.replace(before,after,1);assert modified.replace(after,before,1)==e
compile(modified,str(D/'screen.py'),'exec')
with (D/'screen.py').open('x') as f:f.write(modified)
with (D/'engine_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(e.splitlines(True),modified.splitlines(True),fromfile=str(ENGINE),tofile=str(D/'screen.py'))))
once(D/'engine_inverse.json',dict(parent=pin(ENGINE),engine=pin(D/'screen.py'),before=before,after=after,exact_inverse=True))
retained=BASE.parent/'retained_component.json'
assert retained.exists()
proposal=H/'exact_plane_reuse_source_proposal_v1.json'
assert pin(proposal)['sha256']=='009fe4884a10ec053e03e57a0f897955416077ff41cba5f3f25bd001d025203d'
protocol=dict(schema='exact-plane-reuse-v1',root_selected=True,parent=pin(BASE),scope='native2_only',
 parent_retention=pin(retained),parent_retention_requirement=str(retained),proposal=pin(proposal),
 parent_failed_valid40=pin(BASE.parent/'broad_assessment.json'),
 selection='Parent native8 retained; its original fixed40 misses1%bankgain and remains failed. One complementary exact-cost bundle, not relabeling that40 or changing its gate.',
 intervention='Retain independent cov32/sub32 from active full-lower _flms_cov_star atL1–14 only, preserve original Cpre64/result64 and patch sub32 diagonal from original64 diagonal. Return references explicitly to unique caller; publish Cpre32 after existing layerclear. Direct FP32 CPresponse scatter preserves separate exact64 diagonal for both diagonal consumers. Born add explicitly executes public float64 ufunc into originalFP32out, preserving all64diagonal calculations.',
 proof='Existing Cpre64 offdiagonals widen cov32 exactly; sub64 offdiagonals widen sub32 exactly and both diagonals store same64source. CPresponse is already32; old widen/scatter/narrow returns samebits, while separate diagonal retains originalFP64values. Explicit float64 bornadd uses identical operand conversion and loop, then same32rounding formerly done by _ps32; authoritative64diagonal stays separate. No arithmetic/reduction reassociation.',
 ownership='cov and sub independent fresh outputs, not reused scratcharena. Locals reset eachiteration; Cpre32 is published only after _E2_LAYER.clear with existing Cpre object identity. Original full64 covariance and star result remain available for all64consumers. No new global/persistent state; existing finally/drop semantics unchanged. L0 covariance and terminal covariance paths unchanged. Full-lower currentlineage; fallbackstar stillcasts normally.',
 resources='Same640M UBCcap for candidate and exact4ec comparator; historicalcontrols512M. Original7.5GB processRSS,zeroSwap,setup4.1 and resource/serving gates unchanged.',
 gates='N2 incrementalgain>=.02%,2wins,maxreg0,ALL16bytes andfinite/resources. Ownboundary thenN8gain>=.02%,8wins,maxreg0,CIupper<0,additional6negative,ALL16bytes. FinalbankVALID40gain>=1%+serving; parent failed40 preserved.',
 price=json.loads(proposal.read_text())['prices'],conservative_mean_forecast_pct=.056785236613494876,
 pricing_limit='Forecast uses savedS34shape counts; descendantactualbill mustconfirm.200000F allowance coversO(n)diagonal bookkeeping. No accuracy or bankgain inferred.',
 parity_layers=list(range(16)),parent_native2=pin(BASE.parent/'native2_assessment.json'),parent_native8=pin(BASE.parent/'native8_assessment.json'),
 model_executed=False,serving_qualified=False,submitted=False)
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
c['provenance'] += [pin(retained),pin(proposal),pin(BASE.parent/'native8_assessment.json'),pin(BASE.parent/'broad_assessment.json')]
c['interpretation']=protocol['gates']
c['parent_retention_required']=str(retained)
c['parent6_pinned']='artifacts' in base['blocks']['6']
c.pop('additional_root_review',None)
assert c['prospective_scope']=='native2_only'
once(D/'native_screen.json',c)
s34=copy.deepcopy(c);s34['arms']['base']=json.loads((T.parent/'native_screen_s34_report.json').read_text())['arms']['base']
for size in ('2','8'):
 s34['stages'][size]['result']=str(D/f'native{size}_s34_report.json');s34['stages'][size]['assessment_guard']=str(D/f'native{size}_s34_report_guard.json')
s34['interpretation']='Descriptive S34 report only; incremental gates use actual retained4ec parent in native_screen.json.'
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
