"""Skip only the physically disabled BD j1 projection; stdlib source build."""
from pathlib import Path
import ast,copy,difflib,hashlib,json,shlex
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent;R=P.parents[3]
BASE=H/'hill_climb_s34_pair_read7_scan_v1/candidate.py'
T=H/'hill_climb_combo_radial_metric14_v1/native_screen.json'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert pin(BASE)['sha256']=='2212448b123d4be2dadf9c4c8642b49ad1cb3e073a1cc49684e4fc083a19c14f'
s=BASE.read_text();t=ast.parse(s);f=[x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='_bd_update'][-1]
old=ast.get_source_segment(s,f)
start=old.index('    projection = ')
end=old.index('    _e2_hold.append(fnp.multiply(D, c[:, None, None]')
before=old[start:end]
after="""    if _BD_J1_PHYS is None:
        projection = _e2_H(fnp.sum(_e2_H(bp * D), axis=1)) / denominator
        _e2_hold.append(fnp.subtract(D, _g5_bcast3(bp, projection), out=D))
    else:
        # Physical j1 has zero projection in the parent; evaluate only j0 and j2.
        _bj_p0 = _e2_H(fnp.sum(_e2_H(bp * D0), axis=0)) / denominator
        _bj_p2 = _e2_H(fnp.sum(_e2_H(bp * D2), axis=0)) / denominator
        _e2_hold.append(fnp.subtract(D0, _e2_H(bp * _bj_p0), out=D0))
        _e2_hold.append(fnp.subtract(D2, _e2_H(bp * _bj_p2), out=D2))
        _j1m, _g5_j1tail, _g5_j1col = _g5_j1((1.0, 0.0, 1.0), _BD_J1_PHYS)
        c = (c * _j1m + _g5_j1tail) if c.ndim == 1 else (_g5_mulrow(c, _j1m) + _g5_j1col)
"""
assert '_g5_mulrow(projection, _j1m)' in before and before.count('fnp.subtract(D,')==1
new=old[:start]+after+old[end:];candidate=s.replace(old,new,1)
assert candidate.replace(new,old,1)==s
nt=ast.parse(candidate);compile(candidate,str(D/'candidate.py'),'exec')
assert len(t.body)==len(nt.body)
assert [x.name for x,y in zip(t.body,nt.body) if ast.dump(x)!=ast.dump(y)]==['_bd_update']
(D/'candidate.py').write_text(candidate)
(D/'source_diff.patch').write_text(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
protocol=dict(schema='combo-bd-j1-projection-elision-v1',root_selected=True,parent=pin(BASE),scope='native2_only',
 mechanism='When BD j1 is physical its computed projection is multiplied by zero. Skip that projection and zero subtraction; preserve original j0/j2 operations, c forcing, full D sum and physical raw legs. Retain original fallback when BD_J1_PHYS is None.',
 hypothesis='Identical finite numerical values; row-wise reductions and signed zeros require actual full16 byte-parity confirmation. No prior parity claim.',
 forecast=dict(dominant_saved_flops=8*1024**2*15,approx_bill_saved_pct=.0471,extra_small_calls=True,new_persistent_storage=False),
 gates='Exact-component N2: >=.02% adjusted gain,2wins,maxreg0,all16 byte parity; N8>=.02%,8wins,CIupper<0,additional6negative,all16 parity. Own boundary beforeN8. Final combined bank>=1%VALID40/fullserving.',
 no_coefficient_layer_or_projection_rescue=True,combo_integration='If independently admitted, combine with E4/E6 GNN energy closure and existing retained combo; validate combined result.')
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),protocol=pin(D/'protocol.json'),source_diff=pin(D/'source_diff.patch'),source_inverse_exact=True,changed_parent_functions=['_bd_update'],before=before,after=after,model_executed=False))
c=json.loads(T.read_text())
def rewrite(x):
 if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
 if isinstance(x,list):return [rewrite(v) for v in x]
 return x.replace(T.parent.name,D.name) if isinstance(x,str) else x
c=rewrite(c);c['arms']['candidate']['source']=pin(D/'candidate.py')
c['parity_layers']=list(range(16))
c['provenance']=[pin(p) for p in (BASE,T,D/'build.py',D/'build.json',D/'protocol.json',D/'source_diff.patch',BASE.parent/'native8_assessment.json')]
c['interpretation']=protocol['mechanism']+' Prospective exact-component score and all16 byte-parity gates.'
for stage,n in [('2',2),('8',8)]:
 g=c['stages'][stage]['gates'];g.update(adjusted_pct_at_most=-.02,wins_min=n,max_per_network_regression_pct=0.0)
once(D/'native_screen.json',c)
native=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python';guard=P/'codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py';light=R/'.venv/bin/python'
def wrap(record,sec,rss,cmd):return [str(light),'-B',str(guard),'--record',str(record),'--seconds',str(sec),'--rss-limit',str(rss),*cmd]
b=c['arms']['candidate']['blocks']['2'];evalcmd=[str(native),'-B',b['driver']['path'],str(D/'candidate.py'),','.join(map(str,b['ranks'])),b['tag']]
stage=c['stages']['2']
commands=dict(native2=wrap(b['guard'],b['limits']['seconds'],b['limits']['rss_bytes'],evalcmd),assess_native2=wrap(stage['assessment_guard'],stage['assessment_limits']['seconds'],stage['assessment_limits']['rss_bytes'],[str(native),'-B',c['engine']['path'],str(D/'native_screen.json'),pin(D/'native_screen.json')['sha256'],'2']))
once(D/'commands.json',commands)
print(json.dumps({k:pin(D/k) for k in ['candidate.py','build.json','protocol.json','native_screen.json','commands.json']}))

