"""Real recursive CP host under the unchanged true-previous-m/C protocol."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
import positive_path_birth_reference_v1 as changed
import radial_compact_feed_fullpair_v1 as parent

HERE=Path(__file__).resolve().parent;OUT=HERE/'positive_path_recursive_r35_l12_v1.json';ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
assert json.loads((HERE/'cumulative_positive_path_source_r35_l12_v1.json').read_text())['cumulative_source_gate']
f=np.load(HERE/'residual_one_high_inputs_r35_l12_v1.npz')
W=np.load(HERE/'rows_current_transport_shadow_v2/row_0035/weights.npy')
t=np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
mt=t['mc_post'].astype(np.float64);ct=t['mc_post_cov'].astype(np.float64)
saved={};start=time.time()
class Done(Exception):pass


def run(module,enabled):
    def hook(l,m,C,K21,aux,w):return mt[l]-m if l>=1 else np.zeros_like(m)
    def cov_hook(l,C):return ct[l].copy()
    def post_probe(l,v):
        if l!=12:return
        p=v['p'].astype(np.float64);lam=v['lam'].astype(np.float64);star=v['star'].astype(np.float64)
        b=v['b'];B=v['Bpost'];wn=f['next_weight'];idx=f['indices'];pp=wn.T@p
        K=(pp[idx]**2*lam)@pp.T
        S=wn.T@star;V=wn.T
        K+=((2*S[idx]*V[idx])@V.T+(V[idx]**2)@S.T)/3
        Bn=wn.T@B@wn;bn=wn.T@b
        K+=np.diag(Bn)[idx,None]*bn[None,:]+2*Bn[idx]*bn[idx,None]
        saved['plane']=K
        saved['states']=v['states'].copy()
        raise Done
    source=module.source if module is changed else module.PREDICT_SOURCE
    old='path_mode=True):' if module is changed else 'gate_floor=.001):'
    new='path_mode=True, cov_hook=None):' if module is changed else 'gate_floor=.001, cov_hook=None):'
    edits=((old,new),
           ('            m = m+dm','            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
           ('        radial = radial_new','        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
           ('        previous_alive = alive','        post_probe(l,locals())\n        previous_alive = alive'))
    for a,b in edits:
        assert source.count(a)==1;(source:=source.replace(a,b))
    env=dict(module.env,post_probe=post_probe);exec(source,env)
    kw=dict(hook=hook,cov_hook=cov_hook,gate_floor=.01,cov32=True,metric='covariance',collect_states=True)
    if module is changed:kw['path_mode']=enabled
    try:env['predict'](W,**kw)
    except Done:pass
    return saved['plane'].copy(),saved['states'].copy()


with threadpool_limits(limits=4):
    baseline,_=run(parent,False)
    assert np.array_equal(baseline,f['model_next_K21'])
    disabled,_=run(changed,False)
    assert np.array_equal(disabled,baseline)
    value,states=run(changed,True)
    correction=value-baseline
    E={name:np.load(HERE/f'residual_one_high_truth_r35_l12_{name}_v1.npz')['next_K21']-baseline for name in ('A','B')}
    def metrics(weight):
        a=E['A']*weight;b=E['B']*weight;c=correction*weight
        signal=float(np.mean(a*b));dot=float(.5*np.mean((a+b)*c));energy=float(np.mean(c*c))
        return dict(signal_error_ratio=(signal-2*dot+energy)/signal,signal_cosine=dot/np.sqrt(signal*energy),
                    correction_relative_norm=np.sqrt(energy/signal),A_ratio=float(np.sum((a-c)**2)/np.sum(a*a)),B_ratio=float(np.sum((b-c)**2)/np.sum(b*b)))
    result=dict(unweighted=metrics(1.),weighted=metrics(f['consumer_weight']))
v=result['weighted'];passed=v['signal_error_ratio']<=.75 and v['A_ratio']<=.8 and v['B_ratio']<=.8 and result['unweighted']['signal_error_ratio']<=1
np.savez(ART,baseline=baseline,changed=value,correction=correction)
record=dict(rank=35,layer=12,parent_plane_byte_exact=True,disabled_plane_byte_exact=True,
            results=result,recursive_source_gate=bool(passed),states=states,wall_s=time.time()-start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Actual same-size CP birth update through recursive third states, with coherent true previous m/C. FULL next-plane rows, no projection; same two independent8M streams. No fitted readout, raw score, or metered bill.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='states'},indent=2),flush=True)
