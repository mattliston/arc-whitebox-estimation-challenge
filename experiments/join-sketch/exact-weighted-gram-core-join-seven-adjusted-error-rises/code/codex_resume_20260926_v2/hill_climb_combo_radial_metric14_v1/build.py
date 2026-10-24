"""One L14 radial-metric transport on retained combo; stdlib source build only."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, shlex
from symbolic_fixture import prove
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent;R=P.parents[3]
BASE=H/'hill_climb_s34_pair_read7_scan_v1/candidate.py'
T=H/'hill_climb_s34_ynt_zonal3_v1/native_screen.json'
CONTROL=BASE.parent/'native_screen_continuation.json'

def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rewrite(x):
    if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
    if isinstance(x,list):return [rewrite(v) for v in x]
    return x.replace(T.parent.name,D.name) if isinstance(x,str) else x

assert pin(BASE)['sha256']=='2212448b123d4be2dadf9c4c8642b49ad1cb3e073a1cc49684e4fc083a19c14f'
assert pin(T)['sha256']=='442bf11d03eebcb1ae3b6090d2280d57984386573551e665ae4a71f57d8c18a6'
source=BASE.read_text();tree=ast.parse(source)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary' and '_ps_rest_diag' in ast.get_source_segment(source,n)]
assert len(functions)==1;fn=functions[0];old=ast.get_source_segment(source,fn)
before='            radial = radial_new\n'
after=('            radial = radial_new\n'
    "            if l == 14 and metric == 'covariance':\n"
    '                _rmt_den = fnp.trace(C) ** 2 + 2 * fnp.sum(C * C)\n'
    '                radial = radial_new * (denominator / _rmt_den)\n')
assert old.count(before)==1 and source.count(old)==1
assert old.index('denominator =')<old.index('if _EDGE_ENABLED and 8 <= l <= 14:')<old.index(before)
new=old.replace(before,after,1);candidate=source.replace(old,new,1)
assert candidate.replace(new,old,1)==source
nt=ast.parse(candidate);assert len(tree.body)==len(nt.body)
assert [a.name for a,b in zip(tree.body,nt.body) if ast.dump(a)!=ast.dump(b)]==['_predict_binary']
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'symbolic_result.json',dict(**prove(),fixture=pin(D/'symbolic_fixture.py')))
protocol=dict(schema='combo-radial-metric14-v1',root_selected=True,parent=pin(BASE),scope='native2_only',
    intervention='Only final radial assignment at L14 and metric covariance: radial_new * denominator / D(final C), with original association radial_new*(denominator/D).',
    D='trace(C)^2 + 2*sum(C*C)',numerator='Existing denominator before learned edge and GNN covariance/recentering; no new capture.',
    premise='Preserve double trace of the radial fourth-cumulant closure across the subsequent covariance update. This does not claim raw-fourth-moment or exact recentering consistency.',
    unchanged='All current layer mean/C, birth/third target/bfix paths, coefficient1, all other layer radial values, wrappers, kernel routes, caches, setup and resource limits.',
    parity_layers=list(range(15)),coefficient=1,no_window_or_coefficient_rescue=True,
    prior_negative_evidence='Terminal fixed k4-band convention correction raw+.2803%; L14 third-target recenter raw+3.8006%; historical covariance adjoint found radial consumer weak. No accuracy-gain promise.',
    cost=dict(extra_dense_fp64_operations='One multiply and one sum of n*n, each FP64 rate2; trace, scalar power/multiply/add/divide add O(n) work.',
        dominant_paid_flops=4*1024**2,approx_total_bill_pct=.0016,new_gemm=False,new_persistent_storage=False),
    score_policy='N2 incremental vs combo<=-.1%,>=1win,maxreg<=1%; then own boundary before N8<=-.1%,>=6wins,CIupper<0,additional6negative. Broad only total gain vs S34>=.8%; final VALID40 bank>=1%.',
    symbolic_fixture=pin(D/'symbolic_result.json'),model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),protocol=pin(D/'protocol.json'),
    source_diff=pin(D/'source_diff.patch'),source_inverse_exact=True,changed_parent_functions=['_predict_binary'],
    before=before,after=after,symbolic=pin(D/'symbolic_result.json'),model_executed=False))
original=json.loads(T.read_text());c=rewrite(copy.deepcopy(original));cc=json.loads(CONTROL.read_text())
base=copy.deepcopy(cc['arms']['candidate'])
for size,b in base['blocks'].items():
    folder=H/b['tag'];b['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(b['guard']),
        predictions={str(r):pin(folder/f'pred_{r}.npy') for r in b['ranks']})
c['arms']['base']=copy.deepcopy(base);c['arms']['parity_parent']=copy.deepcopy(base)
c['arms']['candidate']['source']=pin(D/'candidate.py');c['arms']['candidate']['setup']=copy.deepcopy(base['setup'])
c['parity_layers']=list(range(15));c['provenance']=[pin(p) for p in (BASE,T,CONTROL,D/'build.py',D/'protocol.json',D/'build.json',D/'source_diff.patch',D/'symbolic_fixture.py',D/'symbolic_result.json',BASE.parent/'native8_assessment.json')]
c['interpretation']='Fixed L14 radial-metric closure transport. Incremental combo score gates and L0–14 exact parity. Native2 only; own boundary before N8. No window/coefficient rescue.'
assert c['prospective_scope']=='native2_only'
for s in ('2','8'):assert c['stages'][s]['gates']==original['stages'][s]['gates']
once(D/'native_screen.json',c)
s34=copy.deepcopy(c);s34['arms']['base']=copy.deepcopy(original['arms']['base'])
for s in ('2','8'):
    s34['stages'][s]['result']=str(D/f'native{s}_s34_report.json')
    s34['stages'][s]['assessment_guard']=str(D/f'native{s}_s34_report_guard.json')
s34['interpretation']='Separate descriptive S34 comparison; incremental admission remains exclusively native_screen.json versus combo.'
once(D/'native_screen_s34_report.json',s34)
native=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
guard=P/'codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py';light=R/'.venv/bin/python'
def wrap(record,sec,rss,cmd):return [str(light),'-B',str(guard),'--record',str(record),'--seconds',str(sec),'--rss-limit',str(rss),*cmd]
b=c['arms']['candidate']['blocks']['2'];evalcmd=[str(native),'-B',b['driver']['path'],str(D/'candidate.py'),','.join(map(str,b['ranks'])),b['tag']]
commands=dict(native2=wrap(b['guard'],b['limits']['seconds'],b['limits']['rss_bytes'],evalcmd))
for name,config in [('assess_native2','native_screen.json'),('report_s34','native_screen_s34_report.json')]:
    z=json.loads((D/config).read_text());s=z['stages']['2'];cmd=[str(native),'-B',z['engine']['path'],str(D/config),pin(D/config)['sha256'],'2']
    commands[name]=wrap(s['assessment_guard'],s['assessment_limits']['seconds'],s['assessment_limits']['rss_bytes'],cmd)
once(D/'commands.json',commands)
with (D/'commands.txt').open('x') as f:
    for name,cmd in commands.items():f.write('# '+name+'\n'+shlex.join(cmd)+'\n\n')
print(json.dumps({n:pin(D/n) for n in ('candidate.py','native_screen.json','native_screen_s34_report.json','build.json','protocol.json','commands.json')}))
