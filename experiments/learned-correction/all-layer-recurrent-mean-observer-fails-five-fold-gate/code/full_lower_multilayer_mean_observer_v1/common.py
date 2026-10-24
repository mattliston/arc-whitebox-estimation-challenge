import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent
PARENT=D/'full_lower_covariance_edge_packed_v1/candidate.py'
PARENT_SHA='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
RUNTIME=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
NATIVE=D/'weight5_reuse_v1/native.py'
STORE=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
TARGETS=P2/'astra_phase2_reexamination_v1/joint_prefix_final_fit_v1/manifest.json'
TRAIN=[36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58,60]
CHECK=[35,100];PANEL=[61,62,64,65,68,69,70,72,74,76,77,78,80,81];CONFIRM=[37,39,42,48,59,63,66,67]
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def verify(p):assert pin(p['path'])==p,p['path']
def seal_check(name='capture_sources_sealed.json'):
    for p in read(HERE/name)['pins']:verify(p)
    assert sha(PARENT)==PARENT_SHA
def weights_path(rank):
    rows=read(STORE/'manifest.json')['roles']['observer_fit']['row_manifests']
    return (STORE/rows[rank]['path']).parent/'weights.npy'
def guarded(label,seconds,args):
    print(dict(starting=label),flush=True)
    watch=HERE/f'{label}_watch.json'
    cmd=[sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(watch),'--seconds',str(seconds),*map(str,args)]
    with (HERE/f'{label}.log').open('x') as f:r=subprocess.run(cmd,cwd=D,stdout=f,stderr=subprocess.STDOUT)
    g=read(watch);assert r.returncode==0 and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,(label,g)
    print(dict(completed=label,elapsed_s=g['elapsed_s'],peak_rss_bytes=g['max_rss_bytes']),flush=True)
    return g
