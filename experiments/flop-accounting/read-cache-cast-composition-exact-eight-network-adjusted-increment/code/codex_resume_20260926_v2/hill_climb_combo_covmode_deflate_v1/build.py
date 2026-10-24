"""Fixed covariance rank-one arithmetic split; source and scalar metadata only."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, shlex

D=Path(__file__).resolve().parent; H=D.parent; P=H.parent; R=P.parents[3]
BASE=H/'hill_climb_s34_pair_read7_scan_v1/candidate.py'
T=H/'hill_climb_combo_radial_metric14_v1/native_screen.json'

def pin(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def once(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2,allow_nan=False); f.write('\n')

assert pin(BASE)['sha256']=='2212448b123d4be2dadf9c4c8642b49ad1cb3e073a1cc49684e4fc083a19c14f'
assert pin(T)['sha256']=='90fde559b8dbaeeb73d8bb026c587a6d46549633779dfdf1c5c4d1ca64819f85'
source=BASE.read_text(); tree=ast.parse(source)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_flms_cov_star']
fn=functions[-1]; old=ast.get_source_segment(source,fn)
new=old.replace('def _flms_cov_star(w, C, A):','def _flms_cov_star(w, C, A, previous_mean=None):',1)
cast='    C32 = C.astype(fnp.float32)\n'
split=('    _cmd_v = None\n'
       '    if previous_mean is not None:\n'
       '        _cmd_m = fnp.asarray(previous_mean, dtype=fnp.float64)\n'
       '        _cmd_z = C @ _cmd_m\n'
       '        _cmd_d = _cmd_m @ _cmd_z\n'
       '        if bool(_cmd_d > 0):\n'
       '            _cmd_v = _cmd_z / fnp.sqrt(_cmd_d)\n'
       '    C32 = (C - fnp.outer(_cmd_v, _cmd_v)).astype(fnp.float32) if _cmd_v is not None else C.astype(fnp.float32)\n')
assert new.count(cast)==1; new=new.replace(cast,split,1)
output='    Cpre = cov.astype(fnp.float64)\n'
addback=(output+'    if _cmd_v is not None:\n'
         '        _cmd_wv = wt.astype(fnp.float64) @ _cmd_v\n'
         '        Cpre = Cpre + fnp.outer(_cmd_wv, _cmd_wv)\n')
assert new.count(output)==1; new=new.replace(output,addback,1)
call='_flms_cov_star(w32, C, star)'
call_new='_flms_cov_star(w32, C, star, m if 1 <= l <= 14 else None)'
assert source.count(call)==1 and source.count(old)==1
candidate=source.replace(old,new,1).replace(call,call_new,1)
assert candidate.replace(call_new,call,1).replace(new,old,1)==source
nt=ast.parse(candidate); assert len(tree.body)==len(nt.body)
assert [a.name for a,b in zip(tree.body,nt.body) if ast.dump(a)!=ast.dump(b)]==['_predict_binary','_flms_cov_star']
compile(candidate,str(D/'candidate.py'),'exec')
# Every old statement except the covariance cast/add-back point survives exactly.
assert new.replace('def _flms_cov_star(w, C, A, previous_mean=None):','def _flms_cov_star(w, C, A):',1).replace(split,cast,1).replace(addback,output,1)==old
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),before_function=old,after_function=new,before_call=call,after_call=call_new))
protocol=dict(schema='combo-covmode-deflation-v1',root_selected=True,parent=pin(BASE),scope='native2_only',
    intervention='Fixed coefficient1 at existing full covariance/star calls L1–14; pass the actual previous mean explicitly. L0 and terminal diagonal path unchanged.',
    formula='z=C@m; d=m@z; for d>0, v=z/sqrt(d), C_res=C-outer(v,v); Cpre=deep_FP32_push(C_res)+outer(float64(w32.T)@v,float64(w32.T)@v).',
    fallback='d<=0 or no in-scope previous mean keeps the original covariance cast and push. No square root or division is evaluated on d<=0.',
    claim='Real-arithmetic subtract/add-back identity for arbitrary C, including indefinite/asymmetric C. No PSD, dominant-mode, byte-parity, or accuracy-improvement claim.',
    arithmetic='z and d use the existing FP64 covariance and explicit FP64 mean. Residual subtraction is FP64 before the existing FP32 cast. Add-back uses FP64 promotion of the same FP32 weights used by the parent covariance push.',
    unchanged='Every star factorization, star product, star reduction, routing, cache, setup, rank, learned parameter, gate, and all other original source bytes.',
    coefficient=1,layers=list(range(1,15)),parity_layers=[0],no_window_coefficient_direction_rescue=True,
    cost=dict(dominant_extra_flops_per_layer=16*1024**2,dominant_extra_flops_per_network=16*14*1024**2,
              approximate_bill_pct=100*(16*14*1024**2)/267228330900.25,
              details='Two FP64 dense matrix-vector products about8n²; two FP64 outers plus residual subtraction and Cpre addition about8n². Scalar/vector and weight-promotion charges are additional O(n²) copy work; no cost-saving claim. Existing FP32 covariance cast remains.',
              no_new_large_gemm=True,no_new_persistent_storage=True,transient_extra_upper_bound_bytes=4*8*1024**2,
              caveat='Scalar first-order forecast only; conditional fallback/downstream numerical paths and serving resources require actual qualification.'),
    prior='Bounded named-file and RESULTS review found no exact mean-directed covariance deflation/add-back trial. Existing covariance-tail approximation and star/rank experiments are distinct. No favorable result is presumed.',
    score_policy='Incremental vs retained combo: N2 gain>=.1%,>=1win,maxreg<=1%,L0 parity. N8 gain>=.1%,>=6wins,CIupper<0,additional6negative. Own boundary before N8; total S34 gain>=.8% before broad; final VALID40 bank>=1% and full serving.',
    model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),changed_parent_functions=['_predict_binary','_flms_cov_star'],source_inverse_exact=True,star_source_unchanged=True,model_executed=False))

original=json.loads(T.read_text()); c=copy.deepcopy(original)
# Exact path-segment substitution, never a substring replacement of paired-control names.
old_name=T.parent.name
def replace(x):
    if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
    if isinstance(x,list):return [replace(v) for v in x]
    if isinstance(x,str) and x.startswith('/'):
        parts=list(Path(x).parts)
        if old_name in parts:parts=[D.name if v==old_name else v for v in parts]
        parts=[D.name+v[len(old_name):] if v.startswith(old_name+'_') else v for v in parts]
        return str(Path(*parts))
    if isinstance(x,str) and x.startswith(old_name+'_'):return D.name+x[len(old_name):]
    return x
c=replace(c); c['arms']['base']=copy.deepcopy(original['arms']['base']);c['arms']['parity_parent']=copy.deepcopy(original['arms']['parity_parent'])
c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=[0]
c['provenance']=[pin(p) for p in (BASE,T,D/'build.py',D/'protocol.json',D/'build.json',D/'source_inverse.json',D/'source_diff.patch',BASE.parent/'native8_assessment.json')]
c['interpretation']='Fixed previous-mean covariance deflation/add-back, incremental against retained combo. Native2 only; no direction/window/coefficient rescue.'
assert c['prospective_scope']=='native2_only'
for stage in ('2','8'):assert c['stages'][stage]['gates']==original['stages'][stage]['gates']
once(D/'native_screen.json',c)
s34=copy.deepcopy(c); s34_parent=json.loads((T.parent/'native_screen_s34_report.json').read_text());s34['arms']['base']=s34_parent['arms']['base']
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
