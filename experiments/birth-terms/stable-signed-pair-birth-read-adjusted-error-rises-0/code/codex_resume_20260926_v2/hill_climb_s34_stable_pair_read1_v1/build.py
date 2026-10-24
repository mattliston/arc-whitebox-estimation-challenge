"""One L1 stable newborn repeated-index read; source/scalar preparation only."""
from pathlib import Path
import ast,copy,difflib,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent
P=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py';T=H/'hill_climb_s34_ynt_zonal3_v1/native_screen.json'
REG=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/lib/python3.11/site-packages/flopscope/_registry.py'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rewrite(x):
    if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
    if isinstance(x,list):return [rewrite(v) for v in x]
    return x.replace(T.parent.name,D.name) if isinstance(x,str) else x
assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(T)['sha256']=='442bf11d03eebcb1ae3b6090d2280d57984386573551e665ae4a71f57d8c18a6'
source=P.read_text();tree=ast.parse(source);lines=source.splitlines(True);edits=[]
push=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_push_bank_shared_newborn' and "_PHYS_HANDLES['newborn_common']" in ast.get_source_segment(source,n)][-1]
main=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary' and '_E2_LAYER.update(mu=mu' in ast.get_source_segment(source,n)]
assert len(main)==1;main=main[0]
old="    directions = _e2_H(w[selected, :]).T\n"
new=old+"    if (_UVP_CUR[0] == 1 and int(old_columns) == 0\n            and int(common.shape[0]) == int(_E2_LAYER['mu'].shape[0])):\n        _E2_LAYER['stable_pair1'] = _sp1_make(common, directions, t, s)\n"
oldread='_cp_read_young_sq(left2, right, lam, live_indices, shared_old)'
newread='_sp1_read(left2, right, lam, live_indices, shared_old, response_indices, l)'
oldfinal='        finally:\n            _W5_PORT_SCOPE = None'
newfinal="        finally:\n            _E2_LAYER.pop('stable_pair1', None)\n            _W5_PORT_SCOPE = None"
for node,replacements in ((push,[(old,new)]),(main,[(oldread,newread),(oldfinal,newfinal)])):
    before=ast.get_source_segment(source,node);after=before
    for a,b in replacements:assert after.count(a)==1;after=after.replace(a,b,1)
    assert source.count(before)==1
    edits.append(dict(before=before,after=after,function=node.name))
patched=source
for edit in edits:patched=patched.replace(edit['before'],edit['after'],1)
inverse=patched
for edit in reversed(edits):inverse=inverse.replace(edit['after'],edit['before'],1)
assert inverse==source
assert len(ast.parse(patched).body)==len(tree.body)
fragment=(D/'fragment.py').read_text();candidate=patched+'\n'+fragment
compile(candidate,str(D/'candidate.py'),'exec')
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(P),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(P),candidate=pin(D/'candidate.py'),fragment=pin(D/'fragment.py'),suffix_separator='\n',edits=edits))
# Apply tariff weights as well as base callbacks; take bills four per output.
registry=REG.read_text()
assert 'Basic slicing: free (view). Fancy indexing: 4*numel(output)' in registry
assert 'Take elements from array along axis. Cost: numel(output).' in registry
n=r=live=births=1024
weights_path=REG.parent/'data/default_weights.json'
weights=json.loads(weights_path.read_text())['weights']
assert weights['take']==4 and weights['getitem']==weights['concatenate']==1
assert all(weights[k]==1 for k in ('add','subtract','multiply','negative','array_equal'))
assert weights['transpose']==0
extra=3*n*births+15*r*births+10*live*births+7*births
price=dict(registry=pin(REG),default_weights=pin(weights_path),gather_tariff_review=pin(H/'exact_gather_tariff_review_v1.json'),extra_formula='3*n*B + 15*R*B + 10*L*B + 7*B',
    units='FP32 real operations; B birth pairs,R response rows,L live rows,n full pushed rows.',
    derivation='Capture:3nB+4B. Read takes:8(R+L)B after operation weight4; polynomial5RB; left concat2RB; rhs2LB+concat2LB; doubled coefficient B; signed-pair negate/equality2B. Subtract original rhs multiply2LB. Existing pp² and input gathers remain paid in both arms.',
    operations=dict(real_fp32_binary='numel(output)',take='4*numel(output)',concatenate='numel(output)',array_equal='numel(input)',basic_slice=0,transpose=0),
    n1024_B1024_R1024_L1024_extra_flops=extra,whole_s34_rank35_pct=100*extra/267405124808,
    gemm='One original R by2B times2B byL product: shape and routing unchanged. No dense arithmetic saving claim.',
    setup_unchanged=True,extra_peak_storage_bound_bytes=96*1024*1024,actual_native_bill_required=True)
once(D/'price.json',price)
protocol=dict(schema='s34-stable-pair-read1-v1',root_selected=True,parent=pin(P),layer=1,
    formula='p_t=u+v,p_s=u−v. Sum lambda*(p_t_i²*p_t_j−p_s_i²*p_s_j)=2lambda*((u_i²+v_i²)*v_j+2u_i*v_i*u_j).',
    capture='u=common+directions*((t+s)/2),v=directions*((t−s)/2), computed before separate FP32 leg addition.',
    scope='Only L1 youngest cohort, zero inherited columns and full pushed output roster. If deadcut cropped the push, unchanged parent read is used.',
    tensor_state='Original rounded pp, diagonal, omitted-radial, future carried legs and all other layers stay source-identical; only this L1 repeated-index plane read uses stable latent factors.',
    lifetime='Existing per-layer _E2_LAYER field, popped before read, cleared at every next layer and in outer prediction finally; no cross-layer/network value cache.',
    approximation='Exact pre-rounding polynomial identity, not byte parity with separately rounded legs. Accuracy is empirical.',
    prior_distinction='Closed exact reassociation used an extra previous-row dense product; firstjoin_difference_rows changed row selection and old-tier metric. Neither preserved pre-leg newborn contrast.',
    adverse_evidence='E29 same-input precision experiment had cancellation65x and diagonal relative error<=6e-7 with nearzero average raw gain; it did not preserve pre-leg contrast. No promised bankable gain.',
    price=pin(D/'price.json'),parity_layers=[0],setup_counts=dict(bound=682,lvfuse=649,psdb_fused=9),
    primitive=pin(D/'probe.py'),roster=[35,100],scope_authority='native2_only',original_approximate_gates=True,
    no_scope_extension_without_real_positive_evidence=True,source_inverse_exact=True,model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),builder=pin(__file__),fragment=pin(D/'fragment.py'),
    primitive=pin(D/'probe.py'),protocol=pin(D/'protocol.json'),price=pin(D/'price.json'),inverse=pin(D/'source_inverse.json'),diff=pin(D/'source_diff.patch'),
    source_inverse_exact=True,changed_parent_functions=[e['function'] for e in edits],model_executed=False))
base=json.loads(T.read_text());c=rewrite(copy.deepcopy(base));c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=[0]
c['provenance']=[pin(p) for p in (P,T,Path(__file__),D/'fragment.py',D/'probe.py',D/'build.json',D/'protocol.json',D/'price.json',D/'source_inverse.json',D/'source_diff.patch',H/'official_bank_S34_v1.json')]
c['interpretation']='One L1 youngest-cohort stable plane read, before separate leg rounding, original-width GEMM; same S34 N2 controls/gates, L0 exact. No cross-layer state or broader authority.'
assert c['arms']['base']==base['arms']['base'] and c['arms']['parity_parent']==base['arms']['parity_parent']
assert c['arms']['candidate']['setup']['counts']==protocol['setup_counts'] and c['prospective_scope']=='native2_only'
assert all(c['stages'][s][k]==base['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
once(D/'native_screen.json',c)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'),build=pin(D/'build.json'),extra_price=price['whole_s34_rank35_pct'])))
