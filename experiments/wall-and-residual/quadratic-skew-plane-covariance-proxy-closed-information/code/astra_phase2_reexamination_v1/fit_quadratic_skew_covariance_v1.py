"""One fixed coherent-8M, six-train/two-heldout physical-column test.

Inputs are frozen before current-target feedback. Existing frozen v2
features and targets must be byte-identical. No changes to parent files.
"""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as base
from quadratic_skew_covariance_v1 import columns

HERE=Path(__file__).resolve().parent
OUT=HERE/'quadratic_skew_covariance_verdict_v1.json'
FOLDER=HERE/'quadratic_skew_covariance_progress_v1'
MODEL=HERE/'quadratic_skew_covariance_model_v1.npz'
assert not OUT.exists() and not FOLDER.exists() and not MODEL.exists()
assert json.loads((HERE/'quadratic_skew_covariance_certificate_v1.json').read_text())['passed']
assert json.loads((HERE/'quadratic_skew_covariance_cost_v1.json').read_text())['passed']
FOLDER.mkdir()
START=time.time()
STORE=HERE.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
MAN=json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
TRAIN=(35,100,150,200,250,300);VALID=(400,500)

source=base.PREDICT_SOURCE
old='gate_floor=.001):';assert source.count(old)==1
source=source.replace(old,'gate_floor=.001, cov_hook=None):')
old='            m = m+dm';assert source.count(old)==1
source=source.replace(old,old+'\n        if cov_hook is not None:\n            C = cov_hook(l,C)')
old='        radial = radial_new';assert source.count(old)==1
source=source.replace(old,old+'\n        if cov_hook is not None:\n            C = cov_hook(l,C)')
env=dict(base.env);exec(source,env)


def checksum(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def make_row(rank):
    t0=time.time()
    W=np.load(STORE/Path(MAN[rank]['path']).parent/'weights.npy',mmap_mode='r')
    truth=np.load(HERE.parent/'public_g19_attribution_v1'/f'mps_cov2_rank{rank:04d}.npz')
    assert int(truth['P'])==8000000
    mean_truth=truth['mc_post'].astype(np.float64)
    post_cov=truth['mc_post_cov'].astype(np.float64)
    old=np.load(HERE/'rows_covariance_innovation_probe_v2'/f'row_{rank:04d}'/'data.npz')
    Gc=np.zeros((3,3));bc=np.zeros(3);ec=0.
    Gv=np.zeros((3,3));bv=np.zeros(3);ev=0.
    cells=[]

    def hook(l,m,C,K21,aux,w):
        nonlocal Gc,bc,ec,Gv,bv,ev
        if 8<=l<=14:
            a=aux['alpha'];p=aux['pdf'];g=aux['gate'];s=aux['sigma']
            skew=aux['pre_k3']/s**3;kurt=aux['pre_k4']/s**4;ms=m/s
            F=np.stack((g,p,a*p,(a*a-1)*p,ms,a*g,skew*g,skew*p,skew*a*p,skew*ms,
                        kurt*g,kurt*p,kurt*a*p,kurt*ms),axis=1).astype(np.float32)
            R=(aux['pre_cov']/np.outer(s,s)).astype(np.float32)
            j=l-8
            assert np.array_equal(F,old['basis'][j]),('input_basis_parity',rank,l,float(np.max(np.abs(F-old['basis'][j]))))
            assert np.array_equal(R,old['rho'][j]),('input_rho_parity',rank,l,float(np.max(np.abs(R-old['rho'][j]))))
            idx=base.env['select'](g,.01)[0]
            # Complete new features and input fingerprints BEFORE current
            # covariance/mean targets participate in this layer's update.
            H,info=columns(aux['pre_cov'],K21,aux['pre_k4'],a*s,m,idx)
            assert H.shape==(3,1024,1024) and np.isfinite(H).all()
            frozen=dict(covariance_sha256=checksum(aux['pre_cov']),skew_plane_sha256=checksum(K21),
                        columns_sha256=checksum(H),host_sha256=checksum(m))
            target=post_cov[l]-C;target=(target+target.T)/2
            assert np.array_equal(target.astype(np.float32),old['target'][j]),('cov_target_parity',rank,l)
            E=old['target'][j].astype(np.float64)
            flat=H.reshape(3,-1);e=E.ravel()
            Gc+=flat@flat.T;bc+=flat@e;ec+=float(e@e)
            wn=old['weight'][j].astype(np.float64)
            response=old['response'][j].astype(np.float64)
            V=np.stack([np.sum(wn*(x@wn),axis=0)*response for x in H])
            value=np.sum(wn*(E@wn),axis=0)*response
            Gv+=V@V.T;bv+=V@value;ev+=float(value@value)
            cells.append(dict(layer=l,inputs_and_target_byte_exact=True,solve=info,
                              input_frozen_before_target=frozen,column_rms=np.sqrt(np.mean(H*H,axis=(1,2))).tolist()))
        return mean_truth[l]-m if l>=1 else np.zeros_like(m)

    def cov_hook(l,C):return post_cov[l].copy()
    env['predict'](W,hook=hook,cov_hook=cov_hook,gate_floor=.01,cov32=True,metric='covariance')
    assert len(cells)==7
    row=dict(rank=rank,Gc=Gc,bc=bc,ec=ec,Gv=Gv,bv=bv,ev=ev)
    np.savez(FOLDER/f'row_{rank:04d}.npz',**row)
    record=dict(rank=rank,cells=cells,wall_s=time.time()-t0,coherent_samples=8000000,
                baseline_covariance_mse=ec/(7*1024**2),baseline_next_mean_response_mse=ev/(7*1024))
    (FOLDER/f'row_{rank:04d}.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(rank=rank,wall_s=record['wall_s'],inputs_and_targets_byte_exact=True)),flush=True)
    return row


def fit(rows,ids):
    G=sum((r['Gc']/r['ec']+r['Gv']/r['ev'])/(2*len(rows)) for r in rows)
    b=sum((r['bc']/r['ec']+r['bv']/r['ev'])/(2*len(rows)) for r in rows)
    ids=np.array(ids);g=G[np.ix_(ids,ids)];b=b[ids]
    scale=np.sqrt(np.diag(g));assert scale.min()>0
    norm=g/np.outer(scale,scale)
    coefficient=np.linalg.solve(norm+.001*np.eye(len(ids)),b/scale)/scale
    full=np.zeros(3);full[ids]=coefficient
    return full,dict(coefficients=full.tolist(),normalization=scale.tolist(),normalized_gram=norm.tolist(),
                     ridge=.001,condition=float(np.linalg.cond(norm+.001*np.eye(len(ids)))))


def score(r,c):
    return dict(covariance_ratio=float((r['ec']-2*c@r['bc']+c@r['Gc']@c)/r['ec']),
                next_mean_ratio=float((r['ev']-2*c@r['bv']+c@r['Gv']@c)/r['ev']))


with threadpool_limits(limits=4):
    train=[make_row(rank) for rank in TRAIN]
    control,ci=fit(train,(0,1));joint,ji=fit(train,(0,1,2))
    np.savez(MODEL,control=control,joint=joint,train=np.array(TRAIN),valid=np.array(VALID))
    freeze=dict(control=ci,joint=ji,model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                validation_rows_not_yet_loaded=True,wall_s=time.time()-START)
    (FOLDER/'fit_frozen_before_validation.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(dict(fit_frozen_before_validation=True,control=ci,joint=ji)),flush=True)
    valid=[make_row(rank) for rank in VALID]
results=[]
for split,rows in (('train',train),('valid',valid)):
    for r in rows:
        ctl=score(r,control);new=score(r,joint)
        passed=new['covariance_ratio']<=.75 and new['next_mean_ratio']<=.75 and new['next_mean_ratio']<=.9*ctl['next_mean_ratio']
        results.append(dict(rank=r['rank'],split=split,control=ctl,joint=new,
                            incremental_value_ratio=new['next_mean_ratio']/ctl['next_mean_ratio'],passed=passed))
result=dict(passed=all(x['passed'] for x in results if x['split']=='valid'),results=results,fit=freeze,
            wall_s=time.time()-START,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            qualification='Coherent8M oracle-previous-state covariance/source information only; no deployable raw/adjusted score, rollout, fitting sweep, native port or submission.')
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
