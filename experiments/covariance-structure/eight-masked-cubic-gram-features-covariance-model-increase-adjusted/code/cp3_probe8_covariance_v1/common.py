from pathlib import Path
import hashlib,json,math
import numpy as np

HERE=Path(__file__).resolve().parent
H2=HERE.parent;P2=H2.parent;ROOT=P2.parents[3]
PARENT=H2/'candidate_old_both_tail_trace_bounded_v6.py'
PARENT_SHA='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
DEPLOY_PARENT=H2/'candidate_thin_star_route_v1.py'
DEPLOY_SHA='25c22c219420c13db8eae758c16b96a322f24b4de5d40efb43890605b4d3b3ee'
OLD=H2/'native_covariance_innovation_fit_v1'
PREV=H2/'recurrent_covariance_observer_v1'
PILOT=H2/'native_covariance_innovation_v1'
TRAIN=(57,68,45,59,3,12);MINIS=TRAIN
LAYERS=tuple(range(8,15));UPPER=np.triu_indices(1024,1)
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return h.hexdigest()
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def once(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def verify(name='seal.json'):
    assert sha(PARENT)==PARENT_SHA and sha(DEPLOY_PARENT)==DEPLOY_SHA
    for e in read(HERE/name)['sources']:assert pin(e['path'])==e,e['path']
def checked_npz(entry,keys=None):
    assert pin(entry['path'])==entry
    with np.load(entry['path'],allow_pickle=False) as z:
        return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def tables():
    base={x['mini']:x for x in read(PREV/'capture/complete.json')['rows']}
    refs={x['mini']:x for x in read(OLD/'references.json')['rows']}
    return base,refs
def unpack(x):
    a=np.zeros((1024,1024),dtype=x.dtype);a[UPPER]=x;return a+a.T
def project(E,m):
    h=m/np.linalg.norm(m);d=1-2*h*h;assert np.min(d)>0
    A=(E+E.T)*.5;np.fill_diagonal(A,0)
    b=A@h;v=h/d;u=2*(b/d-v*np.dot(h,b/d)/(1+np.dot(h,v)))
    Q=A-(np.outer(u,h)+np.outer(h,u))*.5;np.fill_diagonal(Q,0)
    assert np.isfinite(Q).all() and np.max(abs(Q@h))<1e-10
    return Q
def guard_ok(name):
    g=read(HERE/(name+'_guard.json'))
    assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows']
    assert g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<7_500_000_000
    return g
_FEATURE_TABLE=None
def feature_entry(mini,layer):
    global _FEATURE_TABLE
    if _FEATURE_TABLE is None:
        complete=read(HERE/'feature/complete.json');assert complete['passed'] and complete['completed']
        assert complete['parent_sha256']==PARENT_SHA
        for entry in complete['guards']:
            assert pin(entry['path'])==entry
            g=read(entry['path'])
            assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows']
            assert g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<7_500_000_000
        rows=complete['rows'];assert len(rows)==42
        _FEATURE_TABLE={(r['mini'],r['layer']):r for r in rows}
        assert set(_FEATURE_TABLE)=={(m,l) for m in TRAIN for l in LAYERS}
        assert all(r['shared_fields_bytes_exact'] for r in rows)
    return _FEATURE_TABLE[(mini,layer)]['packet']
