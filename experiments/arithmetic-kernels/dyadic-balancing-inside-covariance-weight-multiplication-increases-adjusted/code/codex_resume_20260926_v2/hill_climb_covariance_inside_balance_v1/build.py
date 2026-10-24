"""Source-only fixed covariance first-product dyadic inside scaling."""
from pathlib import Path
import ast,difflib,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent
BASE=H/'hill_climb_pair_rr_inverse_v1/candidate.py';PRIOR=H/'hill_climb_combo_bd_balance_young_v1'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,o):
 with p.open('x') as f:json.dump(o,f,indent=2,allow_nan=False);f.write('\n')
assert pin(BASE)['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
s=BASE.read_text();t=ast.parse(s)
node=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_flms_cov_star'][-1]
before=ast.get_source_segment(s,node);anchor='    first = _mm(C32, w)'
assert before.count(anchor)==1 and 'retain32=False' in before
after=before.replace(anchor,'    first = _dyadic_covariance_first(C32, w)',1)
suffix='\n'+(D/'fragment.py').read_text();candidate=s.replace(before,after,1)+suffix
assert candidate[:-len(suffix)].replace(after,before,1)==s
nt=ast.parse(candidate);compile(candidate,str(D/'candidate.py'),'exec')
changed=[x.name for x,y in zip(t.body,nt.body[:len(t.body)]) if ast.dump(x)!=ast.dump(y)]
assert changed==['_flms_cov_star'] and len(nt.body)==len(t.body)+1
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source.diff').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),edits=[dict(before=before,after=after)],suffix=suffix,exact_inverse=True,changed_parent_functions=changed))
protocol=dict(schema='covariance-first-dyadic-inside-v1',root_selected=True,parent=pin(BASE),scope='source-only preparation of one fixed native2 trial',
 intervention='Only first=_mm(C32,w) in the final active _flms_cov_star: all14 hidden L1-14 calls. L0 and terminal covariance diagonal helper unchanged. No other products/router edits.',
 formula='an=max(abs(C32),axis0);bn=max(abs(w),axis1). Positive norms: e=clip(rint(0.5*log2(bn/an)),-32,32);otherwise e=0. d=exp2(e).astype(float32). Compute _mm(C32*d[None,:],w/d[:,None]). Real identity(AD)(D^-1B)=AB; floating recursion changes.',
 rationale='Root cited Ballard2016 Sec6.2 inside scaling. Covariance column heterogeneity vs nearly uniform W row norms motivates this hypothesis; no saved operand-range measurement or guaranteed benefit. Current square route usesD7, whose source reports about2.06-2.2x product rounding.',
 reference='https://www.cs.cornell.edu/~arb/papers/fast-matmul-simax2016.pdf',
 arithmetic_limit='Power-of-two scaling is exact in normal range. Exponent clipping alone does not guarantee absence of overflow/subnormal loss. Native finite/accuracy checks are authoritative; no estimator byte-exact claim.',
 ownership='No persistent state/cache/arena. Cached C32 and w not mutated. All preprocessing public paid FNP. Prior-reviewed helper clears_TR_LAST before/after scaled_mm; _W5_PORT_SCOPE disabled only for scaled product and restored in finally. Local held operands retire on return/error.',
 preserved='Same matmul count, shapes, router, second covariance product, star and diagonal logic, learned/physical parameters, cohort/ranks, setup/cache limits.',
 forecast=dict(dense_extra_flops_per_call=6*1024*1024,dense_extra_flops_14_calls=14*6*1024*1024,vector_view_allowance_flops=1000000,network_reference_flops=263000000000,estimated_bill_increase_pct=100*(14*6*1024*1024+1000000)/263000000000,conservative_extra_owned_dense_bytes=4*1024*1024*4,not_measured=True,actual_native_billing_required=True),
 dedup=dict(bounded_sources=['HILL_CLIMB_CURRENT.md','RESULTS.md','fable_engineer_v1/RESULT.md','fable_engineer2_v1/RESULT.md','teammate_covariance_gauge_20260927_v1/RESULT.md','unit_norm_gauge_v1/NEGATIVE_RESULT.md'],covariance_specific_execution_found=False,nearest_failed=pin(PRIOR/'closeout.json'),nearest_scope='Same dyadic rule on inherited-young push FAILED raw+.2659766355%,bill+.1761554488%,adj+.4421872500%,0wins. Original failure preserved.'),
 prospective_parity_layers=[0],gate_policy='Separate root-frozen ordinary approximate native2 screen required; no continuation/bank admission by this source preparation.',no_layer_norm_exponent_coefficient_sweep=True,model_executed=False,numerical_imports=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),fragment=pin(D/'fragment.py'),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source.diff'),prior_helper=pin(PRIOR/'fragment.py'),prior_failed_receipt=pin(PRIOR/'closeout.json'),source_inverse_exact=True,changed_parent_functions=changed,model_executed=False))
print(json.dumps({k:pin(D/k) for k in ('candidate.py','protocol.json','build.json','source_inverse.json')}))
