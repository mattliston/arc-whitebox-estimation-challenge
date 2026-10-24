"""Static source/literal extraction only. Never imports the candidate."""
import ast,base64,hashlib,json,zlib
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
O=D.parents[2]
parents=[O/'engineer/a62_modules/s49'/x/'candidate.py' for x in ('s56k1tw_x','s56ubq1tw_census')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
trees=[ast.parse(p.read_text()) for p in parents]
sources=[p.read_text() for p in parents]
names=('_e2_messages','_e2_wpow','_e2_gnn_consts','gnn_step','_g5_message_edge_base')
proof={}
for name in names:
    nodes=[next(n for n in reversed(t.body) if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
    assert ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False)
    proof[name]=[hashlib.sha256(ast.get_source_segment(s,n).encode()).hexdigest() for s,n in zip(sources,nodes)]
def literal(name):
    return ast.literal_eval(next(n.value for n in trees[0].body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in n.targets)))
g={k:np.frombuffer(zlib.decompress(base64.b64decode(v[1])),np.float32).reshape(v[0]) for k,v in literal('_GNN_LITERAL').items()}
edge={k:np.asarray(v,dtype=np.float64) for k,v in literal('_MESSAGE_EDGE_LITERAL').items()}
assert literal('_MESSAGE_EDGE_ENABLED') is True
A=g['msg_w'][16:32].astype(np.float64)
U,sv,_=np.linalg.svd(A,full_matrices=False)
V=U[:,:2].copy()
for j in range(2):
    if V[np.argmax(np.abs(V[:,j])),j]<0:V[:,j]*=-1
V=V.astype(np.float32)
E=g['enc0_w'][:,46:62].astype(np.float64)
P=V.astype(np.float64)@V.astype(np.float64).T
identity=np.eye(16)
metrics=dict(source_isotropic_retained=float(sum(sv[:2]**2)/sum(sv**2)),singular_values=sv.tolist(),
    encoder_m2_nonzero=int(np.count_nonzero(E)),encoder_m2_frobenius=float(np.linalg.norm(E)),
    encoder_m2_spectral=float(np.linalg.norm(E,2)),projector_orthogonality_error=float(np.linalg.norm(V.T@V-np.eye(2))))
for name,H in [('head',E),('edge_left',edge['left'][16:32].T),('edge_right',edge['right'][16:32].T)]:
    full=float(np.linalg.norm(H@A));residual=float(np.linalg.norm(H@(identity-P)@A))
    metrics[name]=dict(full_source_weighted_frobenius=full,residual_source_weighted_frobenius=residual,
        retained_squared_fraction=1-(residual/full)**2)
baseline=next(n for n in trees[0].body if isinstance(n,ast.FunctionDef) and n.name=='_e2_messages')
with (D/'baseline_e2_messages_v1.py').open('x') as f:f.write(ast.get_source_segment(sources[0],baseline)+'\n')
with (D/'basis_literal_v1.py').open('x') as f:f.write('_SMM_BASIS_LITERAL = '+repr(V.tolist())+'\n')
receipt=dict(status='STATIC_ONLY_NO_MODEL_IMPORT_NO_EVALUATED_DATA',parents=[dict(path=str(p),sha256=sha(p)) for p in parents],
    active_source_same=proof,basis_method='Top two left singular vectors of frozen float32 msg_w[16:32,:], decomposed in float64, largest-absolute coordinate positive, frozen float32 literals.',
    basis_float32_sha256=hashlib.sha256(V.tobytes()).hexdigest(),metrics=metrics,
    reachable_consumers=['gnn_step at layers1–15, signed HM03 before clipping and recurrent state update','active message-edge at layers8–14 through shared _E2_MSG and MSG3 clipping'],
    caveat='Frobenius fractions assume isotropic previous state, linear unclipped consumers; they are not network accuracy predictions.',
    artifact_pins={p.name:sha(p) for p in (D/'baseline_e2_messages_v1.py',D/'basis_literal_v1.py',Path(__file__))})
with (D/'SOURCE_LITERAL_AUDIT_v1.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(D/'SOURCE_LITERAL_AUDIT_v1.json'),sha256=sha(D/'SOURCE_LITERAL_AUDIT_v1.json'),metrics=metrics),indent=2))
