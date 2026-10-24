
from pathlib import Path
import ast, hashlib, json
D=Path(__file__).resolve().parent; H=D.parent
P=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
T=H/'hill_climb_s34_late_c0_95_v1/native_screen.json'
def pin(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2,allow_nan=False); f.write('\n')
def rewrite(x):
    if isinstance(x,dict): return {k:rewrite(v) for k,v in x.items()}
    if isinstance(x,list): return [rewrite(v) for v in x]
    if isinstance(x,str): return x.replace(T.parent.name,D.name)
    return x
assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(T)['sha256']=='25405d726b632f51dbc674d09f69946edfe262b051a91d5d85ff85f9e9fb4a36'
parent=P.read_text(); frag=(D/'fragment.py').read_text(); child=parent+'\n'+frag
ast.parse(child)
with (D/'candidate.py').open('x') as f: f.write(child)
once(D/'protocol.json',dict(baseline=pin(P),fragment=pin(D/'fragment.py'),scope='native2_only',
    operation='D7 k-split leaf-sum for three fixed public dimension classes; private cached views, same arena, existing forward/unwind.',
    shapes=[[1024,2048,1024],[1024,4096,1024],[992,6144,1024]],
    region_floats=[7**7*128]*3,arena_floats=391602176,
    hypothesis='About .31% modeled bill saving before measured rounding effects; model timing is extrapolation only.',
    parity_layers=[0],roster=[35,100],no_bit_exact_claim=True,no_threshold_sweep=True,
    native2_gates=dict(adjusted_pct_at_most=-.1,wins_min=1,max_per_network_regression_pct=1),
    native8_gates=dict(adjusted_pct_at_most=-.1,wins_min=6,ci_upper_below=0,additional6_negative=True),
    final_bank_min_gain_pct=1.0))
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),fragment=pin(D/'fragment.py'),builder=pin(__file__),protocol=pin(D/'protocol.json'),exact_parent_prefix=True))
cfg=rewrite(json.loads(T.read_text()))
cfg['arms']['candidate']['source']=pin(D/'candidate.py')
cfg['arms']['candidate']['setup']['counts']['bound']=686
cfg['arms']['candidate']['setup']['counts']['read7_fused']=4
cfg['parity_layers']=[0]
cfg['provenance']=[pin(p) for p in (P,T,Path(__file__),D/'fragment.py',D/'protocol.json',D/'build.json',H/'official_bank_S34_v1.json')]
cfg['interpretation']='Fixed three-shape depth7 ksplit read, exact S34 prefix; TRAIN2 only, unchanged score/resource gates; layer0 parity. No boundary/broader qualification yet.'
once(D/'native_screen.json',cfg)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'))))

