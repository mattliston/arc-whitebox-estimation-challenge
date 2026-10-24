"""Stdlib-only builder for one fixed star block-Krylov trial."""
from pathlib import Path
import ast, copy, difflib, hashlib, json
D=Path(__file__).resolve().parent;H=D.parent
P=H/'hill_climb_pair_rr_inverse_v1'
T=H/'hill_climb_post_variance_nonnegative_v1'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def once(p,x):
 with Path(p).open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False)+'\n')
parent=pin(P/'candidate.py')
assert parent['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
ret=pin(P/'retained_component.json')
assert ret['sha256']=='6dcd1a0f2b930232f0ef1b43f6f8d039b8acbb0a83067d3aebcecbea857e3aa1'
s=Path(parent['path']).read_text();frag=(D/'fragment.py').read_text()
node=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='_lr_factor'][-1]
lines=s.splitlines(True);before=''.join(lines[node.lineno-1:node.end_lineno])
out=''.join(lines[:node.lineno-1])+frag+''.join(lines[node.end_lineno:])
assert out.replace(frag,before,1)==s
compile(out,str(D/'candidate.py'),'exec')
oldnodes=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)]
newnodes=[n for n in ast.parse(out).body if isinstance(n,ast.FunctionDef)]
assert len(oldnodes)==len(newnodes)
changed=[x.name for x,y in zip(oldnodes,newnodes) if ast.dump(x,include_attributes=False)!=ast.dump(y,include_attributes=False)]
assert changed==['_lr_factor']
once(D/'candidate.py',out)
once(D/'source.diff',''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile=parent['path'],tofile=str(D/'candidate.py'))))
cfg=read(T/'native_screen.json')
for role in ('base','parity_parent'):
 assert cfg['arms'][role]['source']==parent
cfg['arms']['candidate']=copy.deepcopy(cfg['arms']['base'])
cfg['arms']['candidate']['source']=pin(D/'candidate.py')
for size in ('2','6'):
 b=cfg['arms']['candidate']['blocks'][size];b.pop('artifacts',None)
 b['tag']=D.name+'_train'+size+'_v1';b['guard']=str(D/('native'+size+'_guard.json'))
for size in ('2','8'):
 cfg['stages'][size]['result']=str(D/('native'+size+'_assessment.json'))
 cfg['stages'][size]['assessment_guard']=str(D/('native'+size+'_assess_guard.json'))
cfg['boundary_before8']['receipt']=str(D/'NATIVE8_NOT_AUTHORIZED.json')
cfg['boundary_before8']['assessment_guard']=str(D/'NATIVE8_NOT_AUTHORIZED_guard.json')
protocol=dict(schema='fixed-star-block-krylov32-v1',candidate=pin(D/'candidate.py'),source_parent=parent,parent_retained=ret,official_bank=pin(H/'official_bank_S35_v1.json'),formula='Keep Q0=orth(A[:,top32cols]) and Y=A orth(A.T Q0); Q=orth([Q0,Y]); B=Q.T A via two paid existing D3 rank32 products; economy SVD(B), truncate to32, return Q U32 and S32 Vt32.',scope='Only the last active _lr_factor definition. All star callers use the same final rank32 shape. Learned tensors, joins, selection, covariance and source recurrences unchanged.',hypothesis='A larger search subspace can improve the rank32 star approximation without increasing downstream contraction rank. The added factorization cost must be beaten by measured native accuracy.',source_inverse=dict(exact=True,before=before,after=frag),caveats=['This is approximate, not an exact optimization.','Better local Frobenius error does not imply better final means.','No layer/rank/iteration/precision or coefficient variants after a failed fixed screen.','The old shared-old Krylov384 experiment concerns a different operator and join; its failure remains closed.','Two B32 matmuls are separately metered. No paper cost is a bill claim.'],native2_gates=cfg['stages']['2']['gates'],native8_gates=cfg['stages']['8']['gates'],parity_layers=[0],authority='N2 only; own boundary and N8 continuation only after actual N2 PASS; bank requires fully validated >=1% against S35.',model_executed=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),source_parent=parent,fragment=pin(D/'fragment.py'),builder=pin(__file__),protocol=pin(D/'protocol.json'),source_diff=pin(D/'source.diff'),whole_source_inverse=True,only_changed_functions=changed,ast_compile=True))
cfg['provenance']=[pin(D/'build.json'),pin(D/'protocol.json'),ret,pin(P/'native8_assessment.json'),pin(P/'native8_assess_guard.json'),pin(P/'native_screen_continuation.json'),pin(H/'official_bank_S35_v1.json'),pin(T/'native_screen.json'),pin(H/'hill_climb_prefix128_edge_bias_hoist_v1/engine_inverse.json')]
cfg['interpretation']='One fixed star block-Krylov32 accuracy hypothesis versus retained RRPI. Standard approximate N2/N8 gates and resource limits unchanged. No additional variants.'
once(D/'native_screen.json',cfg);cp=pin(D/'native_screen.json')
b=cfg['arms']['candidate']['blocks']['2'];native=str(H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python')
once(D/'commands_native2.json',dict(config=cp,commands=dict(native2=[native,'-B',b['driver']['path'],str(D/'candidate.py'),'35,100',b['tag']],native2_assessment=[native,'-B',cfg['engine']['path'],cp['path'],cp['sha256'],'2']),guards=dict(native2=b['guard'],native2_assessment=cfg['stages']['2']['assessment_guard']),limits=dict(native2_seconds=150,assessment_seconds=120,rss_bytes=7500000000,added_swap_bytes=0),n2_only=True))
print(json.dumps({k:pin(D/n) for k,n in [('candidate','candidate.py'),('config','native_screen.json'),('build','build.json'),('protocol','protocol.json'),('commands','commands_native2.json')]}))
