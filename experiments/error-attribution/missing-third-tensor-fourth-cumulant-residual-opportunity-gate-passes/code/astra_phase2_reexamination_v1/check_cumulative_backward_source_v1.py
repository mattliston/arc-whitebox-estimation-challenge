"""One fixed prefix-source sum; no source/age fitting or intermediate writes."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent
from cumulative_backward_source_v1 import source_slices,certificate
from residual_one_high_oracle_v1 import distinct_plane,repeated_push

HERE=Path(__file__).resolve().parent;OUT=HERE/'cumulative_backward_source_r35_l12_v1.json';ART=OUT.with_suffix('.npz')
assert not OUT.exists() and not ART.exists()
assert json.loads((HERE/'third_error_innovation_r35_l12_v1.json').read_text())['carried_dominated_by_predeclared_gate']
small=certificate();f=np.load(HERE/'residual_one_high_inputs_r35_l12_v1.npz')
W=np.load(HERE/'rows_current_transport_shadow_v2/row_0035/weights.npy')
t=np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
mt=t['mc_post'].astype(np.float64);ct=t['mc_post_cov'].astype(np.float64)
rows=[];cache={};start=time.time()
class Done(Exception):pass


def pre_probe(l,v):
    if l>12:return
    cache.clear();cache.update(star=v['star'].astype(np.float64),W=v['w'].copy(),C=v['Cpre'].copy(),
                              bpre=v['bpre'].copy(),Bpre=v['Bpre'].copy(),radial=float(v['radial']),
                              star_pre=v['star_pre'].copy(),alive=v['alive'].copy())


def hook(l,m,C,K21,aux,w):
    if l<=12:
        if l==12:assert np.array_equal((C+C.T)/2,f['host_covariance'])
        alive=cache['alive'];d=aux['pdf']/aux['sigma']
        cache.update(g=aux['gate']*alive,d=d*alive,e=-aux['alpha']/aux['sigma']*d*alive)
    return mt[l]-m if l>=1 else np.zeros_like(m)


def cov_hook(l,C):return ct[l].copy()


def post_probe(l,v):
    if l<=12:
        rows.append({**{k:val for k,val in cache.items() if k!='alive'},'Bpost':v['Bpost'].copy()})
    if l==12:raise Done


src=parent.PREDICT_SOURCE
for a,b in (
    ('gate_floor=.001):','gate_floor=.001, cov_hook=None):'),
    ('        m, C, target, radial_new, aux = nonlinear','        pre_probe(l,locals())\n        m, C, target, radial_new, aux = nonlinear'),
    ('            m = m+dm','            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
    ('        radial = radial_new','        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
    ('        previous_alive = alive','        post_probe(l,locals())\n        previous_alive = alive')):
    assert src.count(a)==1;src=src.replace(a,b)
env=dict(parent.env,pre_probe=pre_probe,post_probe=post_probe);exec(src,env)
with threadpool_limits(limits=4):
    try:env['predict'](W,hook=hook,cov_hook=cov_hook,gate_floor=.01,cov32=True,metric='covariance')
    except Done:pass
    assert len(rows)==13
    A=np.eye(1024);U=f['U'];total=np.zeros((4,1024,1024));rounding=[];indices=f['indices']
    for l in reversed(range(13)):
        row=rows[l];h=A@U;M=source_slices(**row,H=h)
        total+=np.stack([A.T@m@A for m in M])
        # Target-free selected-row audit of the parent's stored f32 star
        # plane against the exact64 tensor implied by the stored star.
        Kold=row['star'].T.copy()/3;np.fill_diagonal(Kold,np.diag(row['star']))
        exact=repeated_push(Kold,row['W'],indices)
        weight=(row['g'][indices]**2)[:,None]*row['g'][None,:]
        difference=(row['star_pre'][indices]-exact)*weight
        denom=np.linalg.norm(exact*weight)
        rel=float(np.linalg.norm(difference)/max(denom,1e-30));assert rel<1e-5,(l,rel)
        rounding.append(dict(layer=l,relative=rel))
        A=(row['W']*row['g'][None,:])@A
        print(json.dumps(dict(source_layer=l,wall_s=time.time()-start)),flush=True)
    correction=distinct_plane(total,U,f['next_weight'],indices)
    E={name:np.load(HERE/f'residual_one_high_truth_r35_l12_{name}_v1.npz')['next_K21']-f['model_next_K21'] for name in ('A','B')}
    def metrics(weight):
        a=E['A']*weight;b=E['B']*weight;c=correction*weight
        signal=float(np.mean(a*b));dot=float(.5*np.mean((a+b)*c));energy=float(np.mean(c*c))
        return dict(signal_error_ratio=(signal-2*dot+energy)/signal,signal_cosine=dot/np.sqrt(signal*energy),
                    correction_relative_norm=np.sqrt(energy/signal),A_ratio=float(np.sum((a-c)**2)/np.sum(a*a)),B_ratio=float(np.sum((b-c)**2)/np.sum(b*b)))
    result=dict(unweighted=metrics(1.),weighted=metrics(f['consumer_weight']))
v=result['weighted'];gate=v['signal_error_ratio']<=.65 and v['signal_cosine']>=.65 and v['A_ratio']<=.7 and v['B_ratio']<=.7 and result['unweighted']['signal_error_ratio']<=1
np.savez(ART,accumulated_source_slices=total,correction=correction)
record=dict(rank=35,target_layer=12,source_layers=list(range(13)),certificate=small,host_byte_exact=True,
            results=result,cumulative_source_gate=bool(gate),star_plane_rounding=rounding,
            source_slice_norm=float(np.linalg.norm(total)),wall_s=time.time()-start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Fixed natural sum: star/old-trace linear defect, complete Gaussian bond3, current radial metric defect. Exact full-edge backward-directed slices and linear prefix transport; stored f32 plane rounding measured. No source/age selection, intermediate nonlinear/repeated response, state writes, new MC, usable sidecar, metered bill or score.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='star_plane_rounding'},indent=2),flush=True)
