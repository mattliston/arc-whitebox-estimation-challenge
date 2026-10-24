"""One compensation factorization on the pending BD linear candidate; stdlib only."""
from pathlib import Path
import ast,copy,difflib,hashlib,json,shlex
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent;R=P.parents[3]
BASE=H/'hill_climb_combo_bd_j1_skip_v1/candidate.py'
PARENT=H/'hill_climb_combo_bd_linear_v1/candidate.py'
T=PARENT.parent/'native_screen.json'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert pin(BASE)['sha256']=='1078e51058e7674e9a322ebfee10ffa8cb124b70a68b1127cc6011ee274b0dde'
assert pin(PARENT)['sha256']=='3dbed0c626a8c9a12e6a328222f83b1abe1394e8eb9dcd0ec32a3f6e29f4f600'
assert pin(T)['sha256']=='badf2ea1072b83401a59310766b4d129bf5fdd3c68fd160bdf191e88c6959357'
s=PARENT.read_text();tree=ast.parse(s);lines=s.splitlines(True)
f=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_bd_update'][-1]
before=''.join(lines[f.lineno-1:f.end_lineno])
old_delta='    delta = _e2_H(_e2_H(_e2_H(_e2_H(2 * old) * difference) + _e2_H(difference * difference)) * weight) + _e2_H(fnp.einsum(\'i,ji->ij\', quadratic_weight, difference, optimize=False))\n'
new_delta='    delta = _e2_H(_e2_H(_e2_H(new + old) * difference) * weight) + _e2_H(fnp.einsum(\'i,ji->ij\', quadratic_weight, difference, optimize=False))\n    del new\n'
assert before.count(old_delta)==1 and before.count('    del new\n')==1
after=before.replace('    del new\n','',1).replace(old_delta,new_delta,1)
assert s.count(before)==1;candidate=s.replace(before,after,1)
assert candidate.count(after)==1 and candidate.replace(after,before,1)==s
nt=ast.parse(candidate);compile(candidate,str(D/'candidate.py'),'exec')
assert len(tree.body)==len(nt.body)
assert [a.name for a,b in zip(tree.body,nt.body) if ast.dump(a)!=ast.dump(b)]==['_bd_update']
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(PARENT),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(PARENT),before=before,after=after,exact_inverse=True))
review=dict(abs_raw_pct_at_most=.01,applies_stages=[2,8],automatic_continuation=False,native2_incremental_gain_vs_linear_pct_min=.015,linear_native2_reference=pin(PARENT.parent/'native2_assessment.json'))
protocol=dict(schema='combo-bd-linear-delta-v1',root_selected=True,parent=pin(PARENT),comparator=pin(BASE),scope='native2_only',
 intervention='Factor only the stored-FP32 compensation scalar identity in the active _bd_update; preserve new64 until delta is computed, then delete it. All linear fusion, weight, quadratic term, born timing, masks and diagonal clearing stay identical.',
 formula='difference=new64-old64; 2*old64*difference+difference*difference = (new64+old64)*difference in real arithmetic.',
 arithmetic='Different FP64 association; no claim of full byte parity. Source inputs and returned new32 unchanged. new64 lives through delta calculation, adding at most one n*n float64 temporary lifetime (8 MiB at1024), with no persistent state.',parity_layers=[0],
 forecast=dict(additional_saved_flops=4*1024**2*15,approx_additional_s34_bill_saved_pct=4*1024**2*15/267405124808*100,qualification='Two full-plane FP64 operations removed per call. Forecast only; compare native bills and rounding empirically.'),
 selection='Explicit prospective root selection after BD-linear N2 and boundary passed; combine two small algebraic changes once before N8. Fitted/evaluated TRAIN2 evidence is not independent validation.',
 gates='N2 incremental vs retained BD: gain>=.08%,2wins,maxreg<=.1%,L0 parity, abs(raw change)<=.01%, plus root verifies >=.015% increment vs saved linear N2. Own boundary before combined N8 gain>=.08%,>=6wins,CIupper<0,additional6negative,abs(raw change)<=.01%. Total S34>=.8% before broad; final bank>=1% VALID40/fullserving.',
 mandatory_root_review=review,no_coefficient_layer_or_formula_rescue=True,formula_fixture=pin(D/'formula_fixture.py'),parent_retained=pin(BASE.parent/'retained_component.json'),linear_native2=pin(PARENT.parent/'native2_assessment.json'),model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(PARENT),comparator=pin(BASE),builder=pin(__file__),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),formula_fixture=pin(D/'formula_fixture.py'),source_inverse_exact=True,changed_parent_functions=['_bd_update'],model_executed=False))
original=json.loads(T.read_text());old_name=T.parent.name
def rename(x):
 if isinstance(x,dict):return {k:rename(v) for k,v in x.items()}
 if isinstance(x,list):return [rename(v) for v in x]
 if isinstance(x,str) and x.startswith('/'):
  parts=[D.name if v==old_name else D.name+v[len(old_name):] if v.startswith(old_name+'_') else v for v in Path(x).parts];return str(Path(*parts))
 if isinstance(x,str) and x.startswith(old_name+'_'):return D.name+x[len(old_name):]
 return x
c=rename(copy.deepcopy(original));c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=[0]
bdc=json.loads((BASE.parent/'native_screen_continuation.json').read_text());base=copy.deepcopy(bdc['arms']['candidate'])
for size in ('2','6'):
 block=base['blocks'][size];folder=H/block['tag'];block['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(block['guard']),predictions={str(r):pin(folder/f'pred_{r}.npy') for r in block['ranks']})
c['arms']['base']=copy.deepcopy(base);c['arms']['parity_parent']=copy.deepcopy(base)
for size in ('2','8'):
 c['stages'][size]['gates']['adjusted_pct_at_most']=-.08
c['stages']['2']['gates'].update(wins_min=2,max_per_network_regression_pct=.1)
c['provenance']=[pin(p) for p in (BASE,PARENT,PARENT.parent/'build.json',PARENT.parent/'native2_assessment.json',PARENT.parent/'boundary_assessment.json',T,D/'build.py',D/'build.json',D/'protocol.json',D/'source_inverse.json',D/'source_diff.patch',D/'formula_fixture.py',BASE.parent/'retained_component.json',BASE.parent/'native_screen_continuation.json')]
c['interpretation']='One fixed BD linear plus compensation factorization over retained BD. Mandatory abs(raw change)<=.01% at N2/N8 and N2 incremental gain vs saved linear>=.015% root review before continuation; original boundary sentinel remains unavailable.'
c['additional_root_review']=protocol['mandatory_root_review']
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
