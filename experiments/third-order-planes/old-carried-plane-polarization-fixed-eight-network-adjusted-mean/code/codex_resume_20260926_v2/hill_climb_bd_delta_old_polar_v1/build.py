"""Fixed late old-pair polarization and prospective full-eight screening; stdlib builder."""
from pathlib import Path
import ast,copy,difflib,hashlib,json,shlex
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent;R=P.parents[3]
BASE=H/'hill_climb_combo_bd_linear_delta_v1/candidate.py'
PARENT=H/'hill_climb_combo_bd_linear_delta_v1/candidate.py'
T=PARENT.parent/'native_screen.json'
ENGINE=H/'native_screen_engine_v1/screen.py'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert pin(BASE)['sha256']=='6f104e9db5ca32dc5e7724d64e03fd969bfff9d6cb48ce2b55695a998eecd1bb'
assert pin(PARENT)['sha256']=='6f104e9db5ca32dc5e7724d64e03fd969bfff9d6cb48ce2b55695a998eecd1bb'
assert pin(T)['sha256']=='2b0c97146cb387d961c2c786841777e2ba41656346371d2398528a66025bc545'
assert pin(ENGINE)['sha256']=='1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
s=PARENT.read_text();candidate=s;edits=[]
def change(before,after):
 global candidate
 assert candidate.count(before)==1
 candidate=candidate.replace(before,after,1);edits.append(dict(before=before,after=after))
change('    del Pu, Pv\n    return out\n','    _osp_capture(uv, Pu, Pv)\n    del Pu, Pv\n    return out\n')
tree=ast.parse(candidate);lines=candidate.splitlines(True)
f=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_slr_reconstruct'][-1]
before=''.join(lines[f.lineno-1:f.end_lineno])
after=before.replace('    ctx = _OTT_CONTEXT[0]\n','    _osp_begin(U, state)\n    ctx = _OTT_CONTEXT[0]\n',1).replace('    rows, tail = ordered[:kept], ordered[kept:]\n','    rows, tail = ordered[:kept], ordered[kept:]\n    _osp_rows(rows)\n',1)
change(before,after)
change('_cp_read_young_sq(left2, right, lam, live_indices, shared_old)','_osp_read(left2, right, lam, live_indices, shared_old, response_indices)')
fragment=(D/'fragment.py').read_text();candidate+='\n'+fragment
inverse=candidate[:-len('\n'+fragment)]
for edit in reversed(edits):assert inverse.count(edit['after'])==1;inverse=inverse.replace(edit['after'],edit['before'],1)
assert inverse==s
compile(candidate,str(D/'candidate.py'),'exec');ast.parse(candidate)
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(PARENT),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(PARENT),edits=edits,fragment=pin(D/'fragment.py'),exact_inverse=True))
e=ENGINE.read_text();old="        assert math.isfinite(g['adjusted_pct_at_most']) and g['adjusted_pct_at_most'] <= 0"
new="        assert math.isfinite(g['adjusted_pct_at_most']) and (n == '2' or g['adjusted_pct_at_most'] <= 0)"
assert e.count(old)==1;modified=e.replace(old,new,1);assert modified.replace(new,old,1)==e
compile(modified,str(D/'screen.py'),'exec')
with (D/'screen.py').open('x') as f:f.write(modified)
with (D/'engine_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(e.splitlines(True),modified.splitlines(True),fromfile=str(ENGINE),tofile=str(D/'screen.py'))))
once(D/'engine_inverse.json',dict(parent=pin(ENGINE),engine=pin(D/'screen.py'),before=old,after=new,exact_inverse=True,only_change='Allow a prospective positive adjusted threshold at stage2; stage8 nonpositive rule and every other check unchanged.'))
protocol=dict(schema='bd-delta-old-polar-full8-v1',root_selected=True,parent=pin(PARENT),comparator=pin(BASE),scope='native2_only_pending_own_boundary',
 intervention='Fixed L10--14 old-carried pair polarization only. Retain the exact Pu/Pv values before existing rounded plus/minus reconstruction. Assemble Su=4*lambda*Pu*Pv and Dv=2*lambda*(Pu²+Pv²), then the same Dv@Fv2.T, Su@Fu.T and leading-coordinate addition. Young read, pp, squared pp, diagonal, tail/radial repair, current mean and carried CP source stay unchanged until their normal later dependence on changed old off-diagonal read.',
 ownership='One context per prediction/layer with exact state and uv object identities. Explicit full physical row map, composed deadcut then OTT rows, and actual response_indices. Appended zero sentinel preserves padded response rows omitted from reconstruction. Consume each packet exactly once; clear immediately after read and prediction finally. No across-layer/network value cache, inferred ancestry or input mutation.',
 coefficient_identity='Original cohort topology stores plus then minus with equal-and-opposite signed lambda; per-cohort offsets in uv.blocks are preserved. No fitted coefficient or runtime data-dependent sign selection.',
 arithmetic='Same real polynomial of current Pu/Pv and stored signed coefficients, different FP32 path before rounded legs. This is empirical rounding behavior, not an exact-error repair or guaranteed gain. Terminal L15 uses the unchanged fallback.',
 prior_negative='The distinct youngest-only L1 stable read trial failed TRAIN2 (+.662974525% adjusted,0wins). It remains closed. This trial uses late carried u/v factors and a prospectively selected full-eight policy; it does not claim pristine data or proven overfitting.',
 parity_layers=list(range(10)),parity_reason='First changed off-diagonal read is L10; the current learned mean hook also consumes covariance/K21, so only L0--9 outputs are required exact.',
 price='Let m be reconstructed rows, r response rows, P paired columns for each uv segment. Existing preprocessing costs4*r*P FP32 ops. Stable preprocessing costs6*r*P+2*P; paid two concatenations add2*(m+1)*P and two advanced row gathers cost8*r*P (getitem tariff4). Net10*r*P+2*(m+1)*P+2*P plus row-map vectors/views and prediction-local packet lifetime. GEMM shapes/order unchanged. Captured Pu/Pv plus both concatenated copies remain owned until the read finishes (copies alone up to about67MiB for m1024,P8192); total actual RSS is guarded. No measured imbalance or accuracy/speed claim.',
 screening_policy='Fixed original ranks TRAIN2[35,100], remaining6[36,45,52,58,64,70]. For this NEW trial only, stage2 is falsification: mean adjusted<=+1%,maxsingle<=2%,wins>=0 plus all integrity/parity/resource checks. Own boundary before remaining6. Strict pooled N8 gain>=.1%,>=6wins,CIupper<0,additional6negative vs newly retained BD-linear-delta6f104e9d using its genuine saved2+6 controls. This directly evaluates the incremental component on the full8 before broad. Broad total S34>=.8%; final VALID40 bank>=1% and unchanged serving/runtime/resource gates.',
 prospective_policy_not_failed_trial_rescue=True,no_pristine_claim=True,no_window_scale_followons=True,
 baseline_retained=pin(BASE.parent/'retained_component.json'),baseline_native8=pin(BASE.parent/'native8_assessment.json'),parent_native2=pin(PARENT.parent/'native2_assessment.json'),parent_increment_review=pin(PARENT.parent/'root_increment_review.json'),formula_fixture=pin(D/'formula_fixture.py'),model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(PARENT),comparator=pin(BASE),builder=pin(__file__),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),fragment=pin(D/'fragment.py'),formula_fixture=pin(D/'formula_fixture.py'),engine=pin(D/'screen.py'),engine_inverse=pin(D/'engine_inverse.json'),source_inverse_exact=True,model_executed=False))
original=json.loads(T.read_text());old_name=T.parent.name
def rename(x):
 if isinstance(x,dict):return {k:rename(v) for k,v in x.items()}
 if isinstance(x,list):return [rename(v) for v in x]
 if isinstance(x,str) and x.startswith('/'):
  parts=[D.name if v==old_name else D.name+v[len(old_name):] if v.startswith(old_name+'_') else v for v in Path(x).parts];return str(Path(*parts))
 if isinstance(x,str) and x.startswith(old_name+'_'):return D.name+x[len(old_name):]
 return x
c=rename(copy.deepcopy(original));c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=list(range(10));c['engine']=pin(D/'screen.py')
bdc=json.loads((BASE.parent/'native_screen_continuation.json').read_text());base=copy.deepcopy(bdc['arms']['candidate'])
for size in ('2','6'):
 block=base['blocks'][size];folder=H/block['tag'];block['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(block['guard']),predictions={str(r):pin(folder/f'pred_{r}.npy') for r in block['ranks']})
c['arms']['base']=copy.deepcopy(base);c['arms']['parity_parent']=copy.deepcopy(base)
c['stages']['2']['gates'].update(adjusted_pct_at_most=1.0,wins_min=0,max_per_network_regression_pct=2.0)
c['stages']['8']['gates'].update(adjusted_pct_at_most=-.1,wins_min=6)
c['provenance']=[pin(p) for p in (BASE,PARENT,T,ENGINE,D/'build.py',D/'build.json',D/'protocol.json',D/'source_inverse.json',D/'source_diff.patch',D/'fragment.py',D/'formula_fixture.py',D/'screen.py',D/'engine_inverse.json',D/'engine_diff.patch',BASE.parent/'retained_component.json',BASE.parent/'native_screen_continuation.json',PARENT.parent/'native2_assessment.json',PARENT.parent/'root_increment_review.json')]
c['interpretation']=protocol['screening_policy']
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
b=c['arms']['candidate']['blocks']['2'];commands=dict(formula_fixture=[str(light),'-B',str(D/'formula_fixture.py')],native2=wrap(b['guard'],b['limits']['seconds'],b['limits']['rss_bytes'],[str(native),'-B',b['driver']['path'],str(D/'candidate.py'),','.join(map(str,b['ranks'])),b['tag']]))
for name,file in [('assess_native2','native_screen.json'),('report_s34','native_screen_s34_report.json')]:
 cfg=json.loads((D/file).read_text());stage=cfg['stages']['2'];commands[name]=wrap(stage['assessment_guard'],stage['assessment_limits']['seconds'],stage['assessment_limits']['rss_bytes'],[str(native),'-B',cfg['engine']['path'],str(D/file),pin(D/file)['sha256'],'2'])
once(D/'commands.json',commands)
with (D/'commands.txt').open('x') as f:
 for name,command in commands.items():f.write('# '+name+'\n'+shlex.join(command)+'\n\n')
print(json.dumps({name:pin(D/name) for name in ('candidate.py','native_screen.json','build.json','protocol.json','commands.json')}))
