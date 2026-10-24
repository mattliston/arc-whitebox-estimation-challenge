"""Fixed E4-only energies plus admitted BD j1 skip: source-only preparation."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, shlex
D=Path(__file__).resolve().parent; H=D.parent; P=H.parent; R=P.parents[3]
BASE=H/'hill_climb_s34_pair_read7_scan_v1/candidate.py'
T=H/'hill_climb_combo_covmode_deflate_v1/native_screen.json'

def pin(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

assert pin(BASE)['sha256']=='2212448b123d4be2dadf9c4c8642b49ad1cb3e073a1cc49684e4fc083a19c14f'
assert pin(T)['sha256']=='7bc9e843be479b0fd6d3b0f4af7c35f080bea46f130fdf09a4ab75d6ad3926b3'
s=BASE.read_text(); tree=ast.parse(s)
bd=json.loads((H/'hill_climb_combo_bd_j1_skip_v1/build.json').read_text())
assert s.count(bd['before'])==1
s_with_bd=s.replace(bd['before'],bd['after'],1)
old="    E = fnp.matmul(_e2_H(_e2_H(_e2_wpow(W, need6=True))['T135']), k['SQ'])\n"
new="""    _energy_powers = _e2_H(_e2_wpow(W, need6=True))['T135']
    _energy_exact = fnp.matmul(_e2_H(_energy_powers[::2]), _e2_H(k['SQ'][::2]))
    _energy_mass = fnp.sum(_e2_H(_energy_powers[1:2]), axis=2)
    _energy_mean = fnp.mean(_e2_H(k['SQ'][1:2]), axis=1)
    _energy_high = _e2_H(_energy_mass[:, :, None]) * _e2_H(_energy_mean[:, None, :])
    E = fnp.concatenate((_e2_H(_energy_exact[:1]), _energy_high, _e2_H(_energy_exact[1:])), axis=0)
"""
assert s.count(old)==1; candidate=s_with_bd.replace(old,new,1)
assert candidate.count(new)==1 and candidate.replace(new,old,1).replace(bd['after'],bd['before'],1)==s
nt=ast.parse(candidate);assert len(tree.body)==len(nt.body)
assert [a.name for a,b in zip(tree.body,nt.body) if ast.dump(a)!=ast.dump(b)]==['gnn_step','_bd_update']
compile(candidate,str(D/'candidate.py'),'exec')
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),before=old,after=new,exact_inverse=True,changed_functions=['gnn_step','_bd_update'],BD_build=pin(H/'hill_climb_combo_bd_j1_skip_v1/build.json'),BD_before=bd['before'],BD_after=bd['after']))
protocol=dict(schema='combo-energy4-bd-j1-v1',root_selected=True,parent=pin(BASE),scope='native2_only',
    intervention='All existing gnn_step layers1–15: retain the original 3D strided E2/E6 batch with identical per-item layouts; approximate E4 by exact destination weight-power mass times per-channel mean source square. Concatenate unchanged3xn xcm energy layout.',
    coefficient=1,selected_energy_powers=[4],selected_layers=list(range(1,16)),parity_layers=[0],
    formula='T=transpose(W**p), S=corresponding SRC3 squared: Ehat[j,c]=sum_i T[j,i] * mean_i S[i,c], p=4. Exact E2/E6 use T135[::2] @ SQ[::2]; concatenate exact first channel, mean E4, exact second channel in2/4/6 order.',
    preserved='Every destination actual fourth-weight mass; every source-channel mean-square; nonnegativity; exact E2/E6 and all signed W/W2/W3 messages. Same float32 output dtype and3xn xcm shape; W-power construction, SRC square, max/log1p/clip, learned literals and downstream GNN unchanged.',
    uncertainty='This drops covariance between weight powers and squared source. It is exact for constant squared source, not generally exact; recurrent prediction accuracy requires native measurement. Log1p energies have no explicit channel normalization; Gaussian variance2/n scaling is context, not a bound for these networks.',
    first_changed_prediction_layer=1,
    no_further_energy_combination_or_scale_sweep=True,root_selection=pin(D/'root_selection.json'),official_root_selection=pin(H/'energy4_bd_factorial_root_selection_v1.json'),failed_closeouts=[pin(H/'hill_climb_combo_energy46_mean_v1/closeout.json'),pin(H/'hill_climb_combo_energy6_mean_v1/closeout.json')],BD_component_build=pin(H/'hill_climb_combo_bd_j1_skip_v1/build.json'),BD_component_assessment=pin(H/'hill_climb_combo_bd_j1_skip_v1/native2_assessment.json'),
    prior='Explicit TRAIN-informed final channel-ablation trial after E46 and E6 failures, combined with separately N2-passing BD j1. Root selection pins failed protocols/receipts and discloses supersession; no independence or additive-accuracy claim.',
    price=dict(source=pin(H/'hill_climb_s32_bill_census_v1/census.json'),source_only=True,observed_energy_matmul_per_layer=100614144,removed_one_energy_channel_flops_all15=503070720,conservative_energy_extra_flops=16957680,energy_net_saved_flops=486113040,BD_dominant_saved_flops=125829120,forecast_net_saved_flops=611942160,forecast_s34_bill_saving_pct=611942160/267405124808*100,interpretation='Actual dtype-weighted native bill and downstream trajectories decide. No W-power computation saving claimed; no accuracy promise.'),
    score_policy='Incremental vs retained combo: N2 gain>=.1%,>=1win,maxreg<=1%,L0 parity. Own boundary before N8 gain>=.1%,>=6wins,CIupper<0,additional6negative. Broad only S34 total gain>=.8%; final VALID40 bank>=1% plus full serving.',
    formula_fixture=pin(D/'formula_fixture.py'),model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),formula_fixture=pin(D/'formula_fixture.py'),changed_parent_functions=['gnn_step','_bd_update'],source_inverse_exact=True,model_executed=False))

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
c=replace(copy.deepcopy(original));c['arms']['base']=copy.deepcopy(original['arms']['base']);c['arms']['parity_parent']=copy.deepcopy(original['arms']['parity_parent'])
c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=[0]
c['provenance']=[pin(p) for p in (BASE,T,D/'build.py',D/'protocol.json',D/'build.json',D/'source_inverse.json',D/'source_diff.patch',BASE.parent/'native8_assessment.json',D/'formula_fixture.py',D/'root_selection.json',H/'hill_climb_combo_bd_j1_skip_v1/build.json',H/'hill_climb_combo_bd_j1_skip_v1/native2_assessment.json',H/'energy4_bd_factorial_root_selection_v1.json',H/'hill_climb_combo_energy46_mean_v1/closeout.json',H/'hill_climb_combo_energy6_mean_v1/closeout.json')]
c['interpretation']='One TRAIN-informed E4-only closure plus N2-admitted BD j1 skip; incremental against retained combo. Native2 only; no further energy combination or scale sweep.'
assert c['prospective_scope']=='native2_only'
for stage in ('2','8'):assert c['stages'][stage]['gates']==original['stages'][stage]['gates']
once(D/'native_screen.json',c)
s34=copy.deepcopy(c);s34_parent=json.loads((T.parent/'native_screen_s34_report.json').read_text());s34['arms']['base']=s34_parent['arms']['base']
for stage in ('2','8'):
    s34['stages'][stage]['result']=str(D/f'native{stage}_s34_report.json');s34['stages'][stage]['assessment_guard']=str(D/f'native{stage}_s34_report_guard.json')
s34['interpretation']='Descriptive S34 comparison only; incremental admission is native_screen.json against retained combo.'
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
