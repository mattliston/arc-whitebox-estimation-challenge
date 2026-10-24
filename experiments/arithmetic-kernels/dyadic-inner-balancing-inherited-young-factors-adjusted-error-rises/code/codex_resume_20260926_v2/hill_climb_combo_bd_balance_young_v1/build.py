"""Fixed dyadic inner balancing of inherited young pushes; source-only preparation."""
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
anchor='    inherited = _mm(wp, _e2_H(p[:, :old_columns]))'
f=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_push_bank_shared_newborn' and anchor in ast.get_source_segment(s,n)];assert len(f)==1
before=ast.get_source_segment(s,f[0]);assert before.count(anchor)==1
after=before.replace(anchor,'    inherited = _dyadic_young_inherited(wp, _e2_H(p[:, :old_columns]))',1)
suffix='\n'+(D/'fragment.py').read_text();candidate=s.replace(before,after,1)+suffix
assert candidate[:-len(suffix)].replace(after,before,1)==s
nt=ast.parse(candidate);compile(candidate,str(D/'candidate.py'),'exec')
assert len(nt.body)==len(tree.body)+1
assert [a.name for a,b in zip(tree.body,nt.body[:len(tree.body)]) if ast.dump(a)!=ast.dump(b)]==['_push_bank_shared_newborn']
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),before=before,after=after,suffix=suffix,exact_inverse=True,callsite_only_inherited=True))
protocol=dict(schema='combo-bd-dyadic-young-v1',root_selected=True,parent=pin(BASE),scope='native2_only',
 intervention='Only the active inherited _mm(wp,p[:,:old_columns]) in compact young/newborn push; all nonempty inherited widths. Newborn common, direct directions, newborn-none branch and global _mm are unchanged.',
 exponent='an=max(abs(A),axis0);bn=max(abs(B),axis1);valid=(an>0)&(bn>0);invalid norms replaced by1 for the ratio; e=clip(rint(.5*log2(bn/an)),-32,32);d=exp2(e).astype(float32)',
 formula='(A diag(d)) (diag(1/d) B) = A B in real arithmetic. All d are nonzero powers of two; finite normal-range scaling is exact but product recursion rounding changes. Extreme input magnitudes may overflow/underflow despite exponent clipping; native finite checks remain authoritative.',
 unchanged='Signed coefficients, cohort membership, carried CP state, selected ranks, all physical/learned parameters, newborncommon and directions. Source matrices are not modified.',
 ownership='No value cache. Scaled A/B are new paid arrays. _W5_PORT_SCOPE is None only while computing the scaled product and restored in finally. _TR_LAST is cleared before and after to prevent stale shared-A reuse. Loss of original-A reuse is possible and included by full native billing/runtime.',
 price='At least3*(m*k+k*q) paid FP32 element work for abs/max/input scaling, plus O(k) comparisons, selections, log2/exp2(16F each), rounding/clipping/cast; temporary operands also cost memory. No measured CP norm-variability or net saving is claimed.',
 history='Bounded H RESULTS/engineer/engineer2 summaries contain no identical internal-index equilibration test. Checkpoint31 power-of-two checks were CP atom-gauge invariance for a different join metric. Row heterogeneity is structurally plausible, not measured evidence.',
 parity_layers=[0],score_policy='Incremental vs retained BD parent: N2 gain>=.1%,>=1win,maxreg<=1%,L0 parity. Own boundary then N8 gain>=.1%,>=6wins,CIupper<0,additional6negative. S34 total>=.8% before broad; final VALID40 bank>=1% plus full serving.',
 source_only=True,model_executed=False,no_additional_census=True,no_scale_or_scope_sweep=True,BD_parent_retained=pin(BASE.parent/'retained_component.json'))
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),fragment=pin(D/'fragment.py'),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),source_inverse_exact=True,changed_parent_functions=['_push_bank_shared_newborn'],model_executed=False))

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
c=replace(copy.deepcopy(original));bdc=json.loads((BASE.parent/'native_screen.json').read_text());base_arm=copy.deepcopy(bdc['arms']['candidate'])
for block in ('2','6'):
    bb=base_arm['blocks'][block];out=H/bb['tag'];bb['artifacts']=dict(manifest=pin(out/'manifest.json'),rows=pin(out/'rows.jsonl'),guard=pin(bb['guard']),predictions={str(r):pin(out/f'pred_{r}.npy') for r in bb['ranks']})
c['arms']['base']=copy.deepcopy(base_arm);c['arms']['parity_parent']=copy.deepcopy(base_arm)
c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=[0]
c['provenance']=[pin(p) for p in (BASE,T,D/'build.py',D/'protocol.json',D/'build.json',D/'source_inverse.json',D/'source_diff.patch',BASE.parent/'native2_assessment.json',BASE.parent/'native_screen.json',BASE.parent/'retained_component.json',D/'fragment.py')]
c['interpretation']='One fixed dyadic inner balancing trial for inherited young push; incremental against retained BD parent. Native2 only; no scale or scope sweep.'
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
commands=dict(native2=wrap(b['guard'],b['limits']['seconds'],b['limits']['rss_bytes'],cmd))
for name,config in [('assess_native2','native_screen.json'),('report_s34','native_screen_s34_report.json')]:
    z=json.loads((D/config).read_text());stage=z['stages']['2'];cmd=[str(native),'-B',z['engine']['path'],str(D/config),pin(D/config)['sha256'],'2']
    commands[name]=wrap(stage['assessment_guard'],stage['assessment_limits']['seconds'],stage['assessment_limits']['rss_bytes'],cmd)
once(D/'commands.json',commands)
with (D/'commands.txt').open('x') as f:
    for name,cmd in commands.items():f.write('# '+name+'\n'+shlex.join(cmd)+'\n\n')
print(json.dumps({n:pin(D/n) for n in ('candidate.py','native_screen.json','build.json','protocol.json','commands.json')}))
