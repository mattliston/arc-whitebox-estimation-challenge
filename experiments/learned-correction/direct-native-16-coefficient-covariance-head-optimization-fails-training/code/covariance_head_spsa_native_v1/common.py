import ast,copy,hashlib,json,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;D=HERE.parent;P2=D.parent
N=D/'covariance_head_native_response_v1';E=D/'full_lower_covariance_edge_native_v1'
PARENT=D/'full_lower_covariance_edge_packed_v1/candidate.py'
PARENT_SHA='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
RUNTIME=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
NATIVE=D/'weight5_reuse_v1/native.py'
TRAIN=[36,38,40,41,43,44];CHECK=[35,100]
PANEL=[45,46,47,49,50,51,52,53,54,55];CONFIRM=[37,39,42,48,59,63,66,67]

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def verify(p):assert pin(p['path'])==p,p['path']
def seal_check():
    for p in read(HERE/'sources_sealed.json')['pins']:verify(p)
    assert sha(PARENT)==PARENT_SHA

def literal_source():
    source=PARENT.read_text();lines=source.splitlines(keepends=True)
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_EDGE_LITERAL' for t in n.targets))
    old=''.join(lines[node.lineno-1:node.end_lineno]);assert source.count(old)==1
    return source,old,ast.literal_eval(node.value)

def build(path,head):
    import numpy as np
    source,old,literal=literal_source();new_literal=copy.deepcopy(literal)
    head=np.asarray(head,dtype=np.float32);assert head.shape==(16,) and np.isfinite(head).all()
    base=np.asarray(literal['output'],np.float32)
    assert np.linalg.norm(head.astype(np.float64)-base.astype(np.float64))<=1.000001
    new_literal['output']=head.tolist();new='_EDGE_LITERAL = '+repr(new_literal)+'\n'
    changed=source.replace(old,new,1);assert changed.replace(new,old,1)==source
    ast.parse(changed)
    for k in literal:
        expected=head if k=='output' else np.asarray(literal[k],np.float32)
        assert np.asarray(new_literal[k],np.float32).tobytes()==expected.tobytes()
    with Path(path).open('x') as f:f.write(changed)
    return dict(source=pin(path),head=head.tolist(),inverse_source_exact=True,only_output_coefficients_changed=True)

def guarded(label,seconds,args):
    watch=HERE/f'{label}_watch.json';log=HERE/f'{label}.log'
    cmd=[sys.executable,'-B',str(D/'run_guarded.py'),'--record',str(watch),'--seconds',str(seconds),*map(str,args)]
    with log.open('x') as f:r=subprocess.run(cmd,cwd=D,stdout=f,stderr=subprocess.STDOUT)
    g=read(watch)
    assert r.returncode==0 and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0,(label,g)
    return g

def native(label,source,ranks):
    assert sum(len(read(p)['records']) for p in HERE.glob('*/result.json') if 'records' in read(p))+len(ranks)<=120
    seal_check();out=HERE/label
    guarded(label,240 if len(ranks)>1 else 120,[RUNTIME,'-B','-W','ignore',NATIVE,source,out,'--ranks',','.join(map(str,ranks))])
    r=read(out/'result.json');assert r['candidate_sha256']==sha(source) and not r['truth_accessed']
    return r['records']

def qualified(call,parent):
    import numpy as np
    assert sha(call['output'])==call['output_sha256']
    assert call['weights_sha256']==parent['weights_sha256']
    assert call['weight5_cache']['numeric_cache_retired']
    assert call['multiplier']==max(.1,call['flops']/2**41)
    q=np.load(call['output'],allow_pickle=False);p=np.load(parent['output']['path'],allow_pickle=False)
    assert q.shape==p.shape==(16,1024) and q.dtype==p.dtype==np.float32
    assert np.isfinite(q).all() and (q>=0).all() and q[:8].tobytes()==p[:8].tobytes()
    return q

def score(call,truth):
    import numpy as np
    q=np.load(call['output'],allow_pickle=False)
    raw=float(np.mean((q[-1].astype(np.float64)-truth)**2))
    return dict(raw=raw,adjusted=raw*call['multiplier'],flops=call['flops'],multiplier=call['multiplier'])

def target(tp):
    import numpy as np
    verify(tp)
    with np.load(tp['path'],allow_pickle=False) as z:t=z['mc_post'][-1].astype(np.float64)
    assert t.shape==(1024,) and np.isfinite(t).all()
    return t

def aggregate(rows,threshold,wins_required,worst_limit,strict=False):
    pooled=sum(r['candidate']['adjusted'] for r in rows)/sum(r['parent']['adjusted'] for r in rows)
    wins=sum(r['adjusted_ratio']<1 for r in rows);worst=max(r['adjusted_ratio'] for r in rows)
    return dict(rows=rows,pooled_adjusted_ratio=pooled,wins=wins,worst_adjusted_ratio=worst,
        passed=(pooled<threshold if strict else pooled<=threshold) and wins>=wins_required and worst<=worst_limit,
        official_score=False,pristine_holdout=False)
