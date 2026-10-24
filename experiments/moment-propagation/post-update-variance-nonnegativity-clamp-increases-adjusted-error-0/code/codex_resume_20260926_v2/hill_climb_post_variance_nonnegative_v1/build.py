"""One post-hook variance nonnegative trial; stdlib only, no model import."""
from pathlib import Path
import ast,copy,difflib,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent
P=H/'hill_climb_pair_rr_inverse_v1';T=H/'hill_climb_hidden_mean_projection_v1'
NATIVE=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def once(p,x):
 with Path(p).open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False)+'\n')
def check(p):assert pin(p['path'])==p
parent=pin(P/'candidate.py');assert parent['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
ret=pin(P/'retained_component.json');assert ret['sha256']=='6dcd1a0f2b930232f0ef1b43f6f8d039b8acbb0a83067d3aebcecbea857e3aa1'
assert read(ret['path'])['status']=='retained_for_future_bundle'
s=Path(parent['path']).read_text()
before='                _ps_post_cov32 = C'
after='                _ps_Cdiag = fnp.maximum(_ps_Cdiag, 0.0)\n                _ps_post_cov32 = C'
assert s.count(before)==1;out=s.replace(before,after,1);assert out.replace(after,before,1)==s
at=ast.parse(s);bt=ast.parse(out)
def bodies(t):
 return {(n.name,n.lineno):ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
# Compare corresponding top-level functions by source order, ignoring line shifts.
a=[n for n in at.body if isinstance(n,ast.FunctionDef)];b=[n for n in bt.body if isinstance(n,ast.FunctionDef)]
assert len(a)==len(b);changed=[x.name for x,y in zip(a,b) if ast.dump(x,include_attributes=False)!=ast.dump(y,include_attributes=False)]
assert changed==['_predict_binary'];compile(out,str(D/'candidate.py'),'exec')
# Scalar contract only: nonnegative projection leaves nonnegative values unchanged.
cases=0
for value in (-1.0, -1e-12, 0.0, 1e-12, 1.0):
    projected=max(value, 0.0)
    assert projected >= 0.0 and (value < 0.0 or projected == value)
    cases+=1
once(D/'candidate.py',out)
inv=dict(parent=parent,edits=[dict(before=before,after=after)],exact_inverse=True)
once(D/'source.diff',''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile=parent['path'],tofile=str(D/'candidate.py'))))
basecfg=read(P/'native_screen_continuation.json');parent8=pin(P/'native8_assessment.json');n8=read(parent8['path'])
assert n8['admitted'] and n8['all16_parent_bytes_exact'] and n8['candidate']==parent
base=copy.deepcopy(basecfg['arms']['candidate'])
for size in ('2','6'):
 block=base['blocks'][size];folder=H/block['tag']
 block['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(block['guard']),predictions={str(r):pin(folder/f'pred_{r}.npy') for r in block['ranks']})
 for k in ('manifest','rows','guard'):assert block['artifacts'][k]==next(x for x in n8['blocks'] if x['size']==int(size))['arms']['candidate'][k]
 gate_guard=read(block['guard']);assert gate_guard['completed'] and gate_guard['whole_group_retired'] and gate_guard['exit_code']==0 and gate_guard['max_swap_growth_bytes']==0 and not gate_guard['remaining_rows']
cfg=read(T/'native_screen.json')
for role in ('base','parity_parent'):cfg['arms'][role]=copy.deepcopy(base)
cfg['arms']['candidate']=copy.deepcopy(base);cfg['arms']['candidate']['source']=pin(D/'candidate.py')
for size in ('2','6'):
 b=cfg['arms']['candidate']['blocks'][size];b.pop('artifacts',None);b['tag']=D.name+'_train'+size+'_v1';b['guard']=str(D/('native'+size+'_guard.json'))
for size in ('2','8'):
 cfg['stages'][size]['result']=str(D/('native'+size+'_assessment.json'));cfg['stages'][size]['assessment_guard']=str(D/('native'+size+'_assess_guard.json'))
cfg['boundary_before8']['receipt']=str(D/'NATIVE8_NOT_AUTHORIZED.json');cfg['boundary_before8']['assessment_guard']=str(D/'NATIVE8_NOT_AUTHORIZED_guard.json')
protocol=dict(schema='post-hook-variance-nonnegative-single-trial-v1',candidate=pin(D/'candidate.py'),source_parent=parent,parent_retained=ret,parent_native8=parent8,official_bank=pin(H/'official_bank_S35_v1.json'),formula='After existing mean recentering and GN_RC: authoritative variance_i := maximum(authoritative variance_i, 0.0).',scope='Every existing hidden mean-hook call L0-14; exact one-line addition immediately before post_cov32 retention. The terminal final_hook fast path is unchanged.',source_inverse=inv,proof=dict(diagonal_owner='Both original fill_diagonal(C64, _ps_Cdiag) and fill_diagonal(post_cov32, _ps_Cdiag) consume the same projected vector. Existing object identity pair remains valid.',unchanged='All offdiagonal covariance arithmetic, learned mean/GNN state, GN_RC coefficients, radial and source ownership, births and signed atoms remain unchanged.',parity='L0 returned mean is appended after the hook and does not consume the modified post-hook covariance; later recurrence can change.',scalar_contract_cases=cases,scalar_fixture_not_native_exactness=True),caveats=['Nonnegative variance is necessary but does not make the full covariance PSD.','Projection may be inactive or may worsen calibrated closure accuracy; no target-derived selection or extra numerical instrumentation.','The saved pair-map radial quantity and all other closure terms remain unchanged.','No floor/rank/window or coefficient variants if the fixed trial fails.'],native2_gates=cfg['stages']['2']['gates'],native8_gates=cfg['stages']['8']['gates'],parity_layers=[0],authority='N2 only; own boundary and fixed N8 continuation require actual N2 PASS. Final bank requires >=1% vs official S35 and full validation/serving.',model_executed=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),source_parent=parent,source_inverse=inv,protocol=pin(D/'protocol.json'),builder=pin(__file__),source_diff=pin(D/'source.diff'),only_changed_functions=changed,ast_compile=True,scalar_contract_cases=cases))
cfg['provenance']=[pin(D/'build.json'),pin(D/'protocol.json'),ret,parent8,pin(P/'native8_assess_guard.json'),pin(P/'native_screen_continuation.json'),pin(H/'official_bank_S35_v1.json'),pin(T/'native_screen.json'),pin(H/'hill_climb_prefix128_edge_bias_hoist_v1/engine_inverse.json')]
cfg['interpretation']='One nonnegative post-hook variance accuracy hypothesis versus retained RR+PI; approximate N2 gates .1%/one win/maxreg1%, L0 parity. Existing engine and all resource limits unchanged. N8 not authorized until own boundary continuation.'
once(D/'native_screen.json',cfg);cp=pin(D/'native_screen.json')
# Separate S35 comparison is descriptive only, reusing the same candidate outputs.
report=copy.deepcopy(cfg);report['arms']['base']=copy.deepcopy(basecfg['arms']['base']);report['stages']['2']['result']=str(D/'native2_s35_report.json');report['stages']['2']['assessment_guard']=str(D/'native2_s35_report_guard.json');report['interpretation']='Descriptive total versus official S35 using saved outputs; primary selection is RR+PI native2 only.'
report['provenance'].append(cp);once(D/'native_screen_s35_report.json',report);rp=pin(D/'native_screen_s35_report.json')
b=cfg['arms']['candidate']['blocks']['2']
once(D/'commands_native2.json',dict(config=cp,report_config=rp,commands=dict(native2=[str(NATIVE),'-B',b['driver']['path'],str(D/'candidate.py'),'35,100',b['tag']],native2_assessment=[str(NATIVE),'-B',cfg['engine']['path'],cp['path'],cp['sha256'],'2'],s35_report=[str(NATIVE),'-B',cfg['engine']['path'],rp['path'],rp['sha256'],'2']),guards=dict(native2=b['guard'],native2_assessment=cfg['stages']['2']['assessment_guard'],s35_report=report['stages']['2']['assessment_guard']),limits=dict(native2_seconds=150,assessment_seconds=120,rss_bytes=7500000000,added_swap_bytes=0),environment=dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'),'1'),n2_only=True))
print(json.dumps({k:pin(D/n) for k,n in [('candidate','candidate.py'),('config','native_screen.json'),('build','build.json'),('protocol','protocol.json'),('commands','commands_native2.json')]}))
