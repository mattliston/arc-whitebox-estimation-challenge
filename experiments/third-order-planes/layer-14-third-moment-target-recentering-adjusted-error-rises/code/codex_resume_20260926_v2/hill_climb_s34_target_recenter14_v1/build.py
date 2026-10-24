"""One L14 implied-raw-third preservation experiment; source-only preparation."""
from pathlib import Path
import ast,copy,difflib,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent
P=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
T=H/'hill_climb_s34_ynt_zonal3_v1/native_screen.json'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rewrite(x):
    if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
    if isinstance(x,list):return [rewrite(v) for v in x]
    return x.replace(T.parent.name,D.name) if isinstance(x,str) else x
assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(T)['sha256']=='442bf11d03eebcb1ae3b6090d2280d57984386573551e665ae4a71f57d8c18a6'
source=P.read_text();fragment=(D/'fragment.py').read_text()
old='                m = m + dm\n'
new=old+'                if l == 14:\n                    target = _target_recenter14(target, host, _gn_C0, m, C)\n'
assert source.count(old)==1
patched=source.replace(old,new,1)
assert patched.replace(new,old,1)==source
before=ast.parse(source);after=ast.parse(patched)
assert len(before.body)==len(after.body)
changes=[a.name for a,b in zip(before.body,after.body) if ast.dump(a)!=ast.dump(b)]
assert changes==['_predict_binary']
assert "return L['ps_target32'] if L.get('ps_target64') is T else _ps32(T)" in source
candidate=patched+'\n'+fragment
assert candidate[:-len('\n'+fragment)].replace(new,old,1)==source
compile(candidate,str(D/'candidate.py'),'exec')
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(P),tofile=str(D/'candidate.py'))))
protocol=dict(schema='s34-target-recenter14-v1',root_selected=True,parent=pin(P),layer=14,coefficient=1,
    formula='F(m,C)_ij=m_j*C_ii+2*m_i*C_ij+m_i^2*m_j; R3_iij=T_ij+F(m,C). Set Tnew=Told+[F(host,_gn_C0)-F(m,C)].',
    raw_reference='The raw third moment implied by the existing target and mean/covariance immediately before the learned mean hook, including already-applied GN/edge covariance additions.',
    closure_assumption='Preserve that implied raw third moment while accepting the authoritative post-hook mean and partially recentered covariance. GN_RC is unchanged and need not preserve the raw second moment.',
    not_distribution_translation=True,not_claimed_accuracy_repair=True,
    unchanged=['posthook mean','posthook covariance and GN_RC','aux and radial','learned parameters','birth/BD algebra','all source outside one guarded target assignment'],
    state_effect='New target changes L14 innovation/CP birth and residual-star assembly. Existing _ps_target32 identity cache misses this new target and converts it normally.',
    parity_layers=list(range(15)),only_prediction_layer_that_may_change=15,
    cost='FP64 O(n^2) elementwise/outer work at one layer, no GEMM, factorization, training or new arena.',
    duplicate_check='Bounded H study filenames for recenter/recentr/centering/raw_moment/mean_shift and analyst/physicist/porter/H RESULTS for third-target mean recenter: no executed equivalent found. This is not an exhaustive repository search.',
    precedent='Analyst RESULT section38 records covariance-recentering removal +4.00% raw on old VALID16; this supports relevance of centering, not expected gain from this new third-target closure.',
    audits=[pin(H/'teammate_opus_accuracy_followon_audit_v1.md'),pin(H/'teammate_s33_recent_lineage_cost_accuracy_review_v1.md')],
    roster=[35,100],scope='native2_only',gate_unchanged=True,
    no_coefficient_or_window_rescue_if_adverse=True,source_inverse_exact=True,
    setup_counts=dict(bound=682,lvfuse=649,psdb_fused=9),primitive=pin(D/'probe.py'),
    serving_qualified=False,broad_qualified=False,submitted=False,model_executed=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),builder=pin(__file__),fragment=pin(D/'fragment.py'),
    primitive=pin(D/'probe.py'),protocol=pin(D/'protocol.json'),diff=pin(D/'source_diff.patch'),
    source_inverse_exact=True,exact_parent_prefix=False,changed_parent_functions=changes,
    old_statement=old,new_statement=new,numpy_torch_or_model_executed=False))
base=json.loads(T.read_text());c=rewrite(copy.deepcopy(base))
c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=list(range(15))
c['provenance']=[pin(p) for p in (P,T,Path(__file__),D/'fragment.py',D/'probe.py',D/'build.json',D/'protocol.json',D/'source_diff.patch',H/'official_bank_S34_v1.json')]
c['interpretation']='One coefficient1 L14 target recenter preserving the prehook-implied raw third moment while posthook mean/C remain unchanged. Empirical closure test, not exact output repair. Original .1% N2 gates and L0–14 byte parity. No broader authority or rescue sweep.'
assert c['arms']['base']==base['arms']['base'] and c['arms']['parity_parent']==base['arms']['parity_parent']
assert c['arms']['candidate']['setup']['counts']==protocol['setup_counts']
assert c['prospective_scope']=='native2_only'
assert all(c['stages'][s][k]==base['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
once(D/'native_screen.json',c)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'),build=pin(D/'build.json'),protocol=pin(D/'protocol.json'))))
