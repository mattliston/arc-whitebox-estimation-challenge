from pathlib import Path
import ast,copy,difflib,hashlib,json
D=Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_star_consumer_basis32_v1");H=D.parent;P=H/'hill_climb_pair_rr_inverse_v1';T=H/'hill_climb_star_krylov32_v1'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with Path(p).open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False)+'\n')
parent=pin(P/'candidate.py');assert parent['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
s=Path(parent['path']).read_text();frag=(D/'fragment.py').read_text()
lines=s.splitlines(True);edits=[]
for name,old,new in [
 ('_flms_cov_star','Q, B = _lr_factor(A32)','Q, B = _scb32_factor(A32, w)'),
 ('_flms_star_diagonal','Q, B = _lr_factor(A.astype(fnp.float32))','Q, B = _scb32_factor(A.astype(fnp.float32), w)')]:
 n=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
 body=''.join(lines[n.lineno-1:n.end_lineno]);assert body.count(old)==1
 edits.append((n.lineno-1,n.end_lineno,body,body.replace(old,new)))
for start,end,before,after in sorted(edits,reverse=True):lines[start:end]=[after]
out=''.join(lines)+'\n\n'+frag
inv=out[:-len('\n\n'+frag)]
for start,end,before,after in edits:
 assert inv.count(after)==1;inv=inv.replace(after,before,1)
assert inv==s;compile(out,str(D/'candidate.py'),'exec')
once(D/'candidate.py',out)
once(D/'source.diff',''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile=parent['path'],tofile=str(D/'candidate.py'))))
cfg=json.loads((T/'native_screen.json').read_text())
for role in ('base','parity_parent'):assert cfg['arms'][role]['source']==parent
cfg['arms']['candidate']=copy.deepcopy(cfg['arms']['base']);cfg['arms']['candidate']['source']=pin(D/'candidate.py')
for size in ('2','6'):
 b=cfg['arms']['candidate']['blocks'][size];b.pop('artifacts',None);b['tag']=D.name+'_train'+size+'_v1';b['guard']=str(D/('native'+size+'_guard.json'))
for size in ('2','8'):
 cfg['stages'][size]['result']=str(D/('native'+size+'_assessment.json'));cfg['stages'][size]['assessment_guard']=str(D/('native'+size+'_assess_guard.json'))
cfg['boundary_before8']['receipt']=str(D/'NATIVE8_NOT_AUTHORIZED.json');cfg['boundary_before8']['assessment_guard']=str(D/'NATIVE8_NOT_AUTHORIZED_guard.json')
protocol=dict(schema='star-consumer-basis32-v1',source_parent=parent,candidate=pin(D/'candidate.py'),formula='s=sum(w*w,axis1); q=normalize(A@s). Initialize QR([q,top31 energy columns of A]); existing32-wide power Z=QR(A.T Q),Y=A Z; final Q=QR([q,Y[:,1:]]), B=Q.T A. Zero-y uses unchanged parent factorization.',scope='14 hidden plus one terminal star factor; fixed32 directions, one power step, existing D3 routes and consumer arithmetic. Terminal uses its actual selected output-weight columns.',rationale='Contains A@s in exact arithmetic, targeting squared-weight aggregate consumer. Not an optimal weighted SVD or full-tensor exactness claim. Existing additive residual native16 small positive was statistically inconclusive; this is a distinct fixed-rank reallocation hypothesis.',screen='Own N2 raw <= -0.1%, adjusted <= -0.1%, at least1 adjusted win, max regression <=1%, finite16/L0 exact; then existing boundary/N8 gates. No rank/layer/scale/seed sweep or rescue.',official_bank=pin(H/'official_bank_S35_v1.json'),prior_additive_result=pin(H/'hill_climb_star_consumer_residual_v2/RESULT.md'),prior_selector_result=pin(H/'hill_climb_star_residual_pivots_v1/RESULT.md'))
once(D/'protocol.json',protocol)
once(D/'build.json',dict(source_parent=parent,candidate=pin(D/'candidate.py'),fragment=pin(D/'fragment.py'),builder=pin(__file__),protocol=pin(D/'protocol.json'),whole_source_inverse=True,changed_calls=['last _flms_cov_star','last _flms_star_diagonal'],new_helper='_scb32_factor',ast_compile=True))
cfg['provenance']=[pin(D/'build.json'),pin(D/'protocol.json'),pin(P/'retained_component.json'),pin(P/'native8_assessment.json'),pin(P/'native8_assess_guard.json'),pin(P/'native_screen_continuation.json'),pin(H/'official_bank_S35_v1.json'),pin(T/'native_screen.json'),pin(H/'hill_climb_prefix128_edge_bias_hoist_v1/engine_inverse.json')]
cfg['interpretation']='Fixed consumer-constrained rank32 accuracy hypothesis vs RRPI; raw gain gate also must pass. No variants.'
once(D/'native_screen.json',cfg);cp=pin(D/'native_screen.json');b=cfg['arms']['candidate']['blocks']['2']
native=str(H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python')
once(D/'commands_native2.json',dict(config=cp,commands=dict(native2=[native,'-B',b['driver']['path'],str(D/'candidate.py'),'35,100',b['tag']],native2_assessment=[native,'-B',cfg['engine']['path'],cp['path'],cp['sha256'],'2']),guards=dict(native2=b['guard'],native2_assessment=cfg['stages']['2']['assessment_guard']),limits=dict(native2_seconds=150,assessment_seconds=120,rss_bytes=7500000000,added_swap_bytes=0)))
print(json.dumps({n:pin(D/n) for n in ('candidate.py','native_screen.json','protocol.json','commands_native2.json')}))
