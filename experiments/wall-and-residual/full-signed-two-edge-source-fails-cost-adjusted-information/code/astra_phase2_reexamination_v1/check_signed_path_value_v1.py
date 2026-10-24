"""Actual recursive signed source, fixed rank35 pair13 -> mean14 gate."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent
import signed_path_birth_reference_v1 as changed

HERE=Path(__file__).resolve().parent
OUT=HERE/'signed_path_value_r35_l13_v1.json'
ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
assert json.loads((HERE/'signed_path_birth_certificate_v1.json').read_text())['passed']
cost=json.loads((HERE/'signed_path_birth_cost_v1.json').read_text())
W=np.load(HERE/'rows_current_transport_shadow_v2/row_0035/weights.npy')
t=np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
mt=t['mc_post'].astype(np.float64);ct=t['mc_post_cov'].astype(np.float64)
old=np.load(HERE/'rows_covariance_innovation_probe_v2/row_0035/data.npz')
current=np.load(HERE/'current_pair_oracle_attribution_v2/inputs_rank_0035.npz')
start=time.time()

class Done(Exception):pass


def run(module,enabled=False):
    saved={}
    def hook(l,m,C,K21,aux,w):
        if l==13:
            saved.update(mean=m.copy(),cov=C.copy(),K21=K21.copy(),pre_cov=aux['pre_cov'].copy())
            raise Done
        return mt[l]-m if l>=1 else np.zeros_like(m)
    def cov_hook(l,C):return ct[l].copy()
    source=module.source if module is changed else module.PREDICT_SOURCE
    a,b=(('signed_path=True):','signed_path=True, cov_hook=None):')
         if module is changed else ('gate_floor=.001):','gate_floor=.001, cov_hook=None):'))
    edits=((a,b),
           ('            m = m+dm','            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
           ('        radial = radial_new','        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)'))
    for a,b in edits:
        assert source.count(a)==1,a
        source=source.replace(a,b)
    env=dict(module.env);exec(source,env)
    kw=dict(hook=hook,cov_hook=cov_hook,gate_floor=.01,cov32=True,metric='covariance')
    if module is changed:kw['signed_path']=enabled
    try:env['predict'](W,**kw)
    except Done:pass
    assert len(saved)==4
    return saved


with threadpool_limits(limits=4):
    base=run(parent)
    E=ct[13]-base['cov'];E=(E+E.T)/2
    assert np.array_equal(E.astype(np.float32),old['target'][5])
    assert np.array_equal(base['K21'],current['K21'][5])
    disabled=run(changed,False)
    assert all(np.array_equal(base[k],disabled[k]) for k in base)
    new=run(changed,True)
    assert np.array_equal(new['pre_cov'],base['pre_cov'])
    E=old['target'][5].astype(np.float64)
    dc=new['cov']-base['cov']
    wn=old['weight'][5].astype(np.float64)
    resp=old['response'][5].astype(np.float64)
    value=lambda C:np.sum(wn*(C@wn),axis=0)*resp
    v=value(E);dv=value(dc)
    result=dict(covariance_error_ratio=float(np.sum((E-dc)**2)/np.sum(E*E)),
                next_mean_error_ratio=float(np.sum((v-dv)**2)/np.sum(v*v)),
                correction_value_cosine=float(v@dv/np.sqrt((v@v)*(dv@dv))),
                correction_value_relative_norm=float(np.linalg.norm(dv)/np.linalg.norm(v)),
                baseline_one_step_mean_mse=float(np.mean((base['mean']-mt[13])**2)),
                signed_one_step_mean_mse=float(np.mean((new['mean']-mt[13])**2)))
gate=result['covariance_error_ratio']<=.35 and result['next_mean_error_ratio']<=.35
np.savez(ART,**{'parent_'+k:v for k,v in base.items()},**{'signed_'+k:v for k,v in new.items()},delta_cov=dc,delta_value=dv)
record=dict(rank=35,pair_layer=13,value_layer=14,parent_target_byte_exact=True,
            disabled_all_states_byte_exact=True,pre_covariance_byte_exact=True,
            results=result,value_gate=gate,projected_cost_multiplier=cost['records'][0]['projected_multiplier'],
            wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            source_sha256=hashlib.sha256((HERE/'signed_path_birth_reference_v1.py').read_bytes()).hexdigest(),
            qualification='Actual full signed four-column-per-birth recursion, no rank projection or fitted readout. Same original8M previous m/C protocol and covariance target; one fixed qualified diagnostic cell, not independent final MSE or a metered adjusted score.')
OUT.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
