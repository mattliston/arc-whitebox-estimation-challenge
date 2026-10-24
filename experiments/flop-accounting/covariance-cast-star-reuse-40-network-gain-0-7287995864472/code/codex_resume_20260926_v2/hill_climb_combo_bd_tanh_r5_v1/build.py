"""Fixed saturation-root R5 learned-edge activation; source-only preparation."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, shlex
D=Path(__file__).resolve().parent; H=D.parent; P=H.parent; R=P.parents[3]
BASE=H/'hill_climb_combo_bd_j1_skip_v1/candidate.py'
T=H/'hill_climb_combo_covmode_deflate_v1/native_screen.json'

def pin(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

assert pin(BASE)['sha256']=='1078e51058e7674e9a322ebfee10ffa8cb124b70a68b1127cc6011ee274b0dde'
assert pin(T)['sha256']=='7bc9e843be479b0fd6d3b0f4af7c35f080bea46f130fdf09a4ab75d6ad3926b3'
s=BASE.read_text(); tree=ast.parse(s)
f=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_g5_message_edge_base'];assert len(f)==1
before=ast.get_source_segment(s,f[0]);anchor='    _e2_hold.append(fnp.tanh(h,out=h))'
assert before.count(anchor)==1
after=before.replace(anchor,'    _e2_hold.append(_r5_edge_tanh(h))',1)
suffix='\n'+(D/'fragment.py').read_text()
candidate=s.replace(before,after,1)+suffix
assert candidate[:-len(suffix)].replace(after,before,1)==s
nt=ast.parse(candidate);compile(candidate,str(D/'candidate.py'),'exec')
assert len(nt.body)==len(tree.body)+2
assert [a.name for a,b in zip(tree.body,nt.body[:len(tree.body)]) if ast.dump(a)!=ast.dump(b)]==['_g5_message_edge_base']
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),before=before,after=after,suffix=suffix,exact_inverse=True))
protocol=dict(schema='combo-bd-tanh-r5-v1',root_selected=True,parent=pin(BASE),scope='native2_only',
 intervention='Replace only active _g5_message_edge_base FP32 tanh(h,out=h), existing layers8–14, by saturation-root-clipped R5. All other activations, GNN inputs/weights, physical terms and BD j1 remain unchanged.',
 clamp_float32=3.6467385292053223,formula='x=clip(h,-c,c);z=x*x;R5=x*(945+z*(105+z))/(945+z*(420+15*z))',selected_layers=list(range(8,15)),parity_layers=list(range(8)),
 arithmetic='All6 cached scalar constants and both scratch planes explicitlyfloat32. Eleven public paid arithmetic calls/helper use out=; h is already a fresh private activation. Two shape-owned scratch planes are fully overwritten each call and cache holds at most one shape. No inputs or network states are retained.',
 proof=dict(R6='x*(10395+1260*x^2+21*x^4)/(10395+4725*x^2+210*x^4+x^6)',sandwich='R6<=tanh<=R5 for x>=0 by exact ODE residuals +/-x^(10/12)/D^2. R5-R6=x^11/(D5D6), increasing since denominator positive coefficients and degree10.',monotonicity='R5 derivative numerator has positive Bernstein coefficients on x^2 in[0,225/16], so strictly increasing through c.',real_global_error_at_float32_clamp_at_most=.001465346823,standard_fp32_global_allowance=.001467,qualification='FP32 allowance assumes standard correctly rounded float32 arithmetic; no native endpoint accuracy or runtime qualification follows.'),
 price=dict(prior=pin(H/'teammate_edge_tanh_price_v1/pricing.json'),prior_dispatch_review=pin(H/'teammate_edge_tanh_price_v1/fusion_review.json'),old_flops_per_element=16,new_flops_per_element=12,saved_flops_network=469762048,forecast_s34_bill_saving_pct=469762048/267405124808*100,extra_arithmetic_calls_network=70,extra_steady_scratch_bytes=134217728,cache_capacity_shapes=1,cold_extra='Two public empty allocations, one six-element float32 constant array and its views; no private arena or prebind expansion.'),
 dispatch_policy='Root explicitly revises previous self-imposed dispatch-no-worse restriction. Official/local residual, wall, setup, RSS and zeroSwap limits are unchanged; own boundary measurement is mandatory before N8.',
 score_policy='Incremental vs exact BD parent: N2 gain>=.1%,>=1win,maxreg<=1%,L0–7 byte parity. Own boundary then N8 gain>=.1%,>=6wins,CIupper<0,additional6negative. S34 total>=.8% before broad; final VALID40 bank>=1% and full serving.',
 no_scale_window_degree_followons=True,formula_fixture=pin(D/'formula_fixture.py'),BD_parent_native2=pin(BASE.parent/'native2_assessment.json'),BD_parent_retained=pin(BASE.parent/'retained_component.json'),qualification='BD parent is now retained by native8; this candidate remains native2_only and future continuation must bind its existing native6 controls.',model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),fragment=pin(D/'fragment.py'),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),formula_fixture=pin(D/'formula_fixture.py'),source_inverse_exact=True,changed_parent_functions=['_g5_message_edge_base'],model_executed=False))

original=json.loads(T.read_text());old_name=T.parent.name
def replace(x):
    if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
    if isinstance(x,list):return [replace(v) for v in x]
    if isinstance(x,str) and x.startswith('/'):
        parts=list(Path(x).parts)
        parts=[D.name if v==old_name else v for v in parts]
        parts=[D.name+v[len(old_name):] if v.startswith(old_name+'_') else v for v in parts]
        return str(Path(*parts))
    if isinstance(x,str) and x.startswith(old_name+'_'):return D.name+x[len(old_name):]
    return x
c=replace(copy.deepcopy(original));bdc=json.loads((BASE.parent/'native_screen.json').read_text());base_arm=copy.deepcopy(bdc['arms']['candidate']); bb=base_arm['blocks']['2'];out=H/bb['tag'];bb['artifacts']=dict(manifest=pin(out/'manifest.json'),rows=pin(out/'rows.jsonl'),guard=pin(bb['guard']),predictions={str(r):pin(out/f'pred_{r}.npy') for r in bb['ranks']});c['arms']['base']=copy.deepcopy(base_arm);c['arms']['parity_parent']=copy.deepcopy(base_arm)
c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=list(range(8))
c['provenance']=[pin(p) for p in (BASE,T,D/'build.py',D/'protocol.json',D/'build.json',D/'source_inverse.json',D/'source_diff.patch',BASE.parent/'native2_assessment.json',BASE.parent/'native_screen.json',BASE.parent/'retained_component.json',D/'formula_fixture.py',D/'fragment.py')]
c['interpretation']='One fixed R5 learned-edge tanh approximation; incremental against actual BD parent controls. Native2 only; no scale/window/degree follow-ons.'
assert c['prospective_scope']=='native2_only'
for stage in ('2','8'):assert c['stages'][stage]['gates']==original['stages'][stage]['gates']
once(D/'native_screen.json',c)
s34=copy.deepcopy(c);s34_parent=json.loads((T.parent/'native_screen_s34_report.json').read_text());s34['arms']['base']=s34_parent['arms']['base']
for stage in ('2','8'):
    s34['stages'][stage]['result']=str(D/f'native{stage}_s34_report.json');s34['stages'][stage]['assessment_guard']=str(D/f'native{stage}_s34_report_guard.json')
s34['interpretation']='Descriptive S34 comparison only; incremental admission is native_screen.json against actual BD parent.'
once(D/'native_screen_s34_report.json',s34)
native=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python';guard=P/'codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py';light=R/'.venv/bin/python'
def wrap(record,sec,rss,cmd):return [str(light),'-B',str(guard),'--record',str(record),'--seconds',str(sec),'--rss-limit',str(rss),*cmd]
b=c['arms']['candidate']['blocks']['2'];cmd=[str(native),'-B',b['driver']['path'],str(D/'candidate.py'),','.join(map(str,b['ranks'])),b['tag']]
commands=dict(formula_fixture=[str(light),'-B',str(D/'formula_fixture.py')],native2=wrap(b['guard'],b['limits']['seconds'],b['limits']['rss_bytes'],cmd))
for name,config in [('assess_native2','native_screen.json'),('report_s34','native_screen_s34_report.json')]:
    z=json.loads((D/config).read_text());stage=z['stages']['2'];cmd=[str(native),'-B',z['engine']['path'],str(D/config),pin(D/config)['sha256'],'2']
    commands[name]=wrap(stage['assessment_guard'],stage['assessment_limits']['seconds'],stage['assessment_limits']['rss_bytes'],cmd)
once(D/'commands.json',commands)
with (D/'commands.txt').open('x') as f:
    for name,cmd in commands.items():f.write('# '+name+'\n'+shlex.join(cmd)+'\n\n')
print(json.dumps({n:pin(D/n) for n in ('candidate.py','native_screen.json','build.json','protocol.json','commands.json')}))
