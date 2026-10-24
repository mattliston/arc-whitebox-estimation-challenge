"""Fixed first-join projected weighted Gram: source-only candidate preparation."""
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
lines=s.splitlines()
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push'
       and 'Nystrom approximate Gram core' in '\n'.join(lines[n.lineno-1:n.end_lineno])]
assert len(nodes)==1;fn=nodes[0]
old=ast.get_source_segment(s,fn)
patches=[
    ('        K = Y[_jn_rows, :].astype(fnp.float64) if _need_core else None\n',
     '        K = Y[_jn_rows, :].astype(fnp.float64) if _need_core and layer != 7 else None\n'),
    ('        if _need_core:\n', '        if _need_core and layer != 7:\n'),
    ("        Fj = _shared_old384_mm(_e2_H(Qn.T), J)\n",
     "        Fj = _shared_old384_mm(_e2_H(Qn.T), J)\n"
     "        if layer == 7:\n"
     "            _slr_require(old_columns == 0 and _need_core, 'first-join exact core ownership')\n"
     "            core = _shared_old384_mm(Fj, _e2_H(_e2_H(weight[:, None]) * _e2_H(Fj.T)))\n"
     "            core = (core + core.T) * 0.5\n")]
new=old
for before,after in patches:
    assert new.count(before)==1;new=new.replace(before,after,1)
candidate=s.replace(old,new,1);assert candidate!=s and candidate.replace(new,old,1)==s
nt=ast.parse(candidate);assert len(tree.body)==len(nt.body)
assert [a.name for a,b in zip(tree.body,nt.body) if ast.dump(a)!=ast.dump(b)]==['_shared_old384_push']
compile(candidate,str(D/'candidate.py'),'exec')
assert new[:new.index('        _need_core =')]==old[:old.index('        _need_core =')]
assert new[new.index('        joined_core = None\n'):]==old[old.index('        joined_core = None\n'):]
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),before_function=old,after_function=new,patches=[dict(before=a,after=b) for a,b in patches]))
protocol=dict(schema='combo-exact-projected-core7-v1',root_selected=True,parent=pin(BASE),scope='native2_only',
    intervention='Only layer7: skip sampled K construction and R64 solve/core multiply. Keep Y/Q/QR unchanged; after existing Fj=Qn.T@J, store symmetrized FP32 Fj diag(lam[start:stop]^2) Fj.T.',
    coefficient=1,selected_join=7,ownership='Assert old_columns==0 and _need_core at the changed point; actual signed stored coefficients are squared exactly as in the parent.',
    unchanged='Current joining bank, selected rows/order, Y, QR, Qn, Fj projection, all carried factors, all other joins, ranks, learned weights, PP/READ7/scan, caches, setup and resource limits.',
    first_changed_prediction_layer=10,parity_layers=list(range(10)),
    mechanism='Correct first stored Gram to match the actual projected factors. It changes the next join10 sketch/selection; the core is not read by reconstruction or inherited-moment consumers between joins.',
    claim='Exact projected weighted-Gram formula in real arithmetic; its native product uses existing FP32 kernels. No endpoint accuracy or byte-parity claim after layer9.',
    prior='The earlier exact-core recurrence remains immediately before the active Nyström override. S30 adopted cheaper approximate cores with other changes. No current S34 exact-core7 reverse trial was found in bounded named records; earlier joint Gram metric and jitter tests are distinct.',
    price=dict(source=pin(H/'hill_climb_s32_bill_census_v1/census.json'),observed_rank=35,shape=[384,6144,384],removed_core_flops_excluding_unchanged_qr=530546688,
               full_native_d5_model_gram_flops=1018712847.8016001,weight_product_flops=2359296,
               approximate_net_added_flops=491000000,approximate_bill_increase_pct=.184,
               interpretation='First-order saved-shape source forecast; actual dispatch/padding/binding and downstream trajectories require measurement. Raw improvement must exceed roughly.284% to clear a .1% adjusted gate.'),
    no_window_coefficient_join_rescue=True,
    score_policy='Incremental vs retained combo: N2 gain>=.1%,>=1win,maxreg<=1%,L0–9 parity. N8 gain>=.1%,>=6wins,CIupper<0,additional6negative. Own boundary before N8; total S34 gain>=.8% before broad; final VALID40 bank>=1% and full serving.',
    model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),changed_parent_functions=['_shared_old384_push'],source_inverse_exact=True,unchanged_current_join_Q_Y_QR_factors=True,model_executed=False))

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
c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=list(range(10))
c['provenance']=[pin(p) for p in (BASE,T,D/'build.py',D/'protocol.json',D/'build.json',D/'source_inverse.json',D/'source_diff.patch',BASE.parent/'native8_assessment.json')]
c['interpretation']='One fixed join7 projected weighted-Gram core; incremental against retained combo. Native2 only; no join/window/coefficient rescue.'
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
commands=dict(native2=wrap(b['guard'],b['limits']['seconds'],b['limits']['rss_bytes'],cmd))
for name,config in [('assess_native2','native_screen.json'),('report_s34','native_screen_s34_report.json')]:
    z=json.loads((D/config).read_text());stage=z['stages']['2'];cmd=[str(native),'-B',z['engine']['path'],str(D/config),pin(D/config)['sha256'],'2']
    commands[name]=wrap(stage['assessment_guard'],stage['assessment_limits']['seconds'],stage['assessment_limits']['rss_bytes'],cmd)
once(D/'commands.json',commands)
with (D/'commands.txt').open('x') as f:
    for name,cmd in commands.items():f.write('# '+name+'\n'+shlex.join(cmd)+'\n\n')
print(json.dumps({n:pin(D/n) for n in ('candidate.py','native_screen.json','build.json','protocol.json','commands.json')}))
