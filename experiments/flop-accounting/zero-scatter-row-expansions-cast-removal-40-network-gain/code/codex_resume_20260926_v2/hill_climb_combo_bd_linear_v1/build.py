"""One fixed BD physical-path linear fusion on retained BD1078; stdlib only."""
from pathlib import Path
import ast,copy,difflib,hashlib,json,shlex
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent;R=P.parents[3]
BASE=H/'hill_climb_combo_bd_j1_skip_v1/candidate.py'
T=H/'hill_climb_combo_bd_tanh_r5_v1/native_screen.json'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert pin(BASE)['sha256']=='1078e51058e7674e9a322ebfee10ffa8cb124b70a68b1127cc6011ee274b0dde'
assert pin(T)['sha256']=='38376b4943ed4a358347ed963ac8f4b3e029a1390fefb8ae03fca3384ae79ff8'
s=BASE.read_text();tree=ast.parse(s);lines=s.splitlines(True)
f=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_bd_update'][-1]
old=''.join(lines[f.lineno-1:f.end_lineno]);start=old.index('    _s19_raw = None\n');end=old.index('    fnp.fill_diagonal(change, 0.0)\n',start)
before=old[start:end]
fallback=''.join('    '+line if line.strip() else line for line in before.splitlines(True))
after='    if _BD_J1_PHYS is None:\n'+fallback+'''    else:
        # Preserve the unweighted directions until both original projections exist.
        _bl_p0 = _e2_H(fnp.sum(_e2_H(bp * D0), axis=0)) / denominator
        _bl_p2 = _e2_H(fnp.sum(_e2_H(bp * D2), axis=0)) / denominator
        _j1m, _g5_j1tail, _g5_j1col = _g5_j1((1.0, 0.0, 1.0), _BD_J1_PHYS)
        c = (c * _j1m + _g5_j1tail) if c.ndim == 1 else (_g5_mulrow(c, _j1m) + _g5_j1col)
        _bl_projection = _e2_H(_e2_H(c[0]) * _bl_p0) + _e2_H(_e2_H(c[2]) * _bl_p2)
        _bl_effective = c
        if _S19_BD_ADD is not None:
            _bl_raw = fnp.asarray(_S19_BD_ADD[:3], dtype=fnp.float64)
            _bl_effective = c + (_bl_raw if c.ndim == 1 else _e2_H(_bl_raw[:, None]))
        _e2_hold.append(fnp.multiply(D, _bl_effective[:, None, None] if c.ndim == 1 else _bl_effective[:, None, :], out=D))
        change = fnp.sum(D, axis=0)
        _e2_hold.append(fnp.subtract(change, _e2_H(bp * _bl_projection), out=change))
'''
new=old[:start]+after+old[end:];assert s.count(old)==1;candidate=s.replace(old,new,1)
assert candidate.count(after)==1 and candidate.replace(after,before,1)==s
nt=ast.parse(candidate);compile(candidate,str(D/'candidate.py'),'exec')
assert len(tree.body)==len(nt.body)
assert [a.name for a,b in zip(tree.body,nt.body) if ast.dump(a)!=ast.dump(b)]==['_bd_update']
raw_assign=[x.value for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_S19_BD_ADD' for t in x.targets)][-1]
raw=ast.literal_eval(raw_assign);assert raw[1]==0.0 and raw[0]!=0.0 and raw[2]!=0.0
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),before=before,after=after,exact_inverse=True,fallback_verbatim=True))
protocol=dict(schema='combo-bd-linear-v1',root_selected=True,parent=pin(BASE),scope='native2_only',
 intervention='Only the active physical-j1 BD path: preserve unweighted D0/D1/D2 and original pr0/pr2; force c1 identically; form c0*pr0+c2*pr2; add the raw-leg coefficients to c; weight D once; sum and subtract one bp projection. None fallback is the verbatim original block nested under its branch. Remaining diagonal, alive mask and stored-FP32 compensation unchanged.',
 formula='sum_a (c_a+r_a)D_a - bp*(c0*pr0+c2*pr2), where forced c1 is physical and pr0/pr2 use the original shared floored denominator.',
 coefficient_scope='Scalar per direction or column-only coefficients; both layouts handled. No division by coefficients.',
 current_raw_coefficients=raw[:3],parity_layers=[0],
 arithmetic='Same real expression, different FP64 association before stored FP32 conversion. No full byte-parity claim. Internal D scratch is fully overwritten next call; no source input mutation or new persistent state.',
 forecast=dict(dominant_saved_flops=12*1024**2*15,approx_s34_bill_saved_pct=12*1024**2*15/267405124808*100,extra_small_vector_work=True,new_persistent_storage=False,qualification='Source forecast only: six full-plane float64 operations per call removed, less small vector/constant work. Actual native bill decides.'),
 gates='N2 incremental vs BD: gain>=.02%,2wins,maxreg<=.1%,L0 parity, plus mandatory root review abs(raw change)<=.01%. Own boundary before N8 gain>=.02%,>=6wins,CIupper<0,additional6negative. Total S34>=.8% before broad; final bank>=1% VALID40/fullserving.',
 mandatory_native2_raw_review=dict(abs_raw_pct_at_most=.01,automatic_continuation=False),
 no_coefficient_layer_or_formula_rescue=True,formula_fixture=pin(D/'formula_fixture.py'),parent_retained=pin(BASE.parent/'retained_component.json'),model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),formula_fixture=pin(D/'formula_fixture.py'),source_inverse_exact=True,changed_parent_functions=['_bd_update'],model_executed=False))
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
 c['stages'][size]['gates']['adjusted_pct_at_most']=-.02
c['stages']['2']['gates'].update(wins_min=2,max_per_network_regression_pct=.1)
c['provenance']=[pin(p) for p in (BASE,T,D/'build.py',D/'build.json',D/'protocol.json',D/'source_inverse.json',D/'source_diff.patch',D/'formula_fixture.py',BASE.parent/'retained_component.json',BASE.parent/'native_screen_continuation.json')]
c['interpretation']='One fixed BD linear reassociation over retained BD. Additional mandatory abs(raw change)<=.01% root review before continuation; original boundary sentinel remains unavailable.'
c['additional_root_review']=protocol['mandatory_native2_raw_review']
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
