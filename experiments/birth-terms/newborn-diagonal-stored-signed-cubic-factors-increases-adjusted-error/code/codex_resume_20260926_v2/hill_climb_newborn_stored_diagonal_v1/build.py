"""One stored-newborn diagonal source trial; stdlib only, no model import."""
from pathlib import Path
import ast,copy,difflib,hashlib,json
from fractions import Fraction as F
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
before='            _ps_born_diag = _ps_born_diag + fnp.diag(_bd_plane_delta)'
after='''            # Newborn CP diagonal from precisely the rounded, masked stored legs.
            _ns_a32, _ns_t32, _ns_s32 = fnp.unstack(_E2_LAYER['ats32'])
            _ns_lam32 = _ns_a32 if _S19_LAM is None else _ns_a32 * _S19_LAM[1]
            _ns_lam64 = _ns_lam32.astype(fnp.float64)
            _ns_t64, _ns_s64 = _ns_t32.astype(fnp.float64), _ns_s32.astype(fnp.float64)
            _ps_born_diag = (_ns_lam64 * ((_ns_t64 * _ns_t64) * _ns_t64
                                         - (_ns_s64 * _ns_s64) * _ns_s64)) * births
            del _ns_a32, _ns_t32, _ns_s32, _ns_lam32, _ns_lam64, _ns_t64, _ns_s64'''
assert s.count(before)==1;out=s.replace(before,after,1);assert out.replace(after,before,1)==s
at=ast.parse(s);bt=ast.parse(out)
def bodies(t):
 return {(n.name,n.lineno):ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef)}
# Compare corresponding top-level functions by source order, ignoring line shifts.
a=[n for n in at.body if isinstance(n,ast.FunctionDef)];b=[n for n in bt.body if isinstance(n,ast.FunctionDef)]
assert len(a)==len(b);changed=[x.name for x,y in zip(a,b) if ast.dump(x,include_attributes=False)!=ast.dump(y,include_attributes=False)]
assert changed==['_predict_binary'];compile(out,str(D/'candidate.py'),'exec')
# Exact rational tensor identity, with zero base diagonal and opposite stored weights.
cases=0
for selected in (False,True):
 for alive in (False,True):
  for av,tv,sv,bv in ((F(2,3),F(5,4),F(-1,2),F(7,5)),(F(-3,7),F(-2),F(3,2),F(-2,9)),(F(0),F(0),F(0),F(0))):
   lam=av*F(9932375877606482,10**16);t=tv*alive;s_=sv*alive
   stored=selected*(lam*t**3-lam*s_**3)
   formula=selected*lam*(t**3-s_**3)
   assert stored==formula and lam*bv**3-lam*bv**3==0
   if not alive or not selected:assert formula==0
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
protocol=dict(schema='stored-newborn-diagonal-single-trial-v1',candidate=pin(D/'candidate.py'),source_parent=parent,parent_retained=ret,parent_native8=parent8,official_bank=pin(H/'official_bank_S35_v1.json'),formula='D_i = births_i * lambda32_i * (t32_i^3 - s32_i^3), widened to float64 before products; lambda32 = a32 * learned newborn scale with the same float32 multiplication used for stored selected lambda; t32/s32 already include alive.',scope='All existing birth assemblies L0-14, one fixed analytic replacement after BD; preserve offdiagonal born, factors, masks, learned scales, inherited diagonal and all consumers.',source_inverse=inv,proof=dict(zero_base_diagonal='bp is filled with zero; BD change is filled with zero before new32; paired offdiagonal base cubes cancel under opposite weights.',ats32_owner='Original _bd_update populates current a/t/s/alive identity-owned ats32 unconditionally before returning.',mask='births is the exact selected-set mask; t32/s32 already contain alive. Existing live factor remains.',scalar_fraction_cases=cases,scalar_fixture_not_native_exactness=True),caveats=['Learned newborn scale intentionally changed the previous closure; this consistency hypothesis can hurt accuracy.','This is not an exact-parent arithmetic optimization and does not promise all16 parity; L0 output precedes this assembly.','Stored-parameter real tensor identity does not certify finite-precision future CP GEMM/reduction or compressed-star recovery.','Older low-rank star residual repairs are distinct negative evidence; no coefficient/window/scale rescue or grid.'],native2_gates=cfg['stages']['2']['gates'],native8_gates=cfg['stages']['8']['gates'],parity_layers=[0],authority='N2 only; own boundary and fixed N8 continuation require actual N2 PASS. Final bank requires >=1% vs official S35 and full validation/serving.',model_executed=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),source_parent=parent,source_inverse=inv,protocol=pin(D/'protocol.json'),builder=pin(__file__),source_diff=pin(D/'source.diff'),only_changed_functions=changed,ast_compile=True,scalar_fraction_cases=cases))
cfg['provenance']=[pin(D/'build.json'),pin(D/'protocol.json'),ret,parent8,pin(P/'native8_assess_guard.json'),pin(P/'native_screen_continuation.json'),pin(H/'official_bank_S35_v1.json'),pin(T/'native_screen.json'),pin(H/'hill_climb_prefix128_edge_bias_hoist_v1/engine_inverse.json')]
cfg['interpretation']='One stored-newborn diagonal accuracy hypothesis versus retained RR+PI; approximate N2 gates .1%/one win/maxreg1%, L0 parity. Existing engine and all resource limits unchanged. N8 not authorized until own boundary continuation.'
once(D/'native_screen.json',cfg);cp=pin(D/'native_screen.json')
# Separate S35 comparison is descriptive only, reusing the same candidate outputs.
report=copy.deepcopy(cfg);report['arms']['base']=copy.deepcopy(basecfg['arms']['base']);report['stages']['2']['result']=str(D/'native2_s35_report.json');report['stages']['2']['assessment_guard']=str(D/'native2_s35_report_guard.json');report['interpretation']='Descriptive total versus official S35 using saved outputs; primary selection is RR+PI native2 only.'
report['provenance'].append(cp);once(D/'native_screen_s35_report.json',report);rp=pin(D/'native_screen_s35_report.json')
b=cfg['arms']['candidate']['blocks']['2']
once(D/'commands_native2.json',dict(config=cp,report_config=rp,commands=dict(native2=[str(NATIVE),'-B',b['driver']['path'],str(D/'candidate.py'),'35,100',b['tag']],native2_assessment=[str(NATIVE),'-B',cfg['engine']['path'],cp['path'],cp['sha256'],'2'],s35_report=[str(NATIVE),'-B',cfg['engine']['path'],rp['path'],rp['sha256'],'2']),guards=dict(native2=b['guard'],native2_assessment=cfg['stages']['2']['assessment_guard'],s35_report=report['stages']['2']['assessment_guard']),limits=dict(native2_seconds=150,assessment_seconds=120,rss_bytes=7500000000,added_swap_bytes=0),environment=dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'),'1'),n2_only=True))
print(json.dumps({k:pin(D/n) for k,n in [('candidate','candidate.py'),('config','native_screen.json'),('build','build.json'),('protocol','protocol.json'),('commands','commands_native2.json')]}))
