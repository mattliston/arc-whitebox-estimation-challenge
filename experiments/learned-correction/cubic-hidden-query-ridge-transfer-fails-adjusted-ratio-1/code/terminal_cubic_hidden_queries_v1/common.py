import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent;A=P2/'astra_phase2_reexamination_v1'
SOURCE=D/'full_lower_covariance_edge_packed_v1/candidate.py'
PARENT_SHA='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
RUNTIME=A/'openblas_runtime_env_v1/bin/python'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def verify(name='sources_sealed.json'):
    x=read(HERE/name)
    for p in x['pins']:assert sha(p['path'])==p['sha256'],p['path']
    return x
def weights(rank):
    m=read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests'];return (STORE/m[rank]['path']).parent/'weights.npy'
def truth(rank):return P2/f'public_g22_operator_features_v1/truthpost/truthpost_rank{rank:04d}.npz'
def guarded(name,seconds,args):
    with (HERE/f'{name}.log').open('x') as f:
        code=subprocess.run([sys.executable,str(D/'run_guarded.py'),'--record',str(HERE/f'{name}_watch.json'),'--seconds',str(seconds),*map(str,args)],stdout=f,stderr=subprocess.STDOUT).returncode
    r=read(HERE/f'{name}_watch.json');assert code==0 and r['completed'] and r['whole_group_retired'] and r['max_swap_growth_bytes']==0,(name,r)
    print(json.dumps(dict(stage=name,completed=True,rss=r['max_rss_bytes'],wall_s=r['elapsed_s'])),flush=True)
