"""Fixed original-cube synthetic certificate, never a P2 score/candidate."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
import whole_pair_kernel_core_v1 as core


HERE = Path(__file__).resolve().parent
OUT = HERE / "whole_pair_kernel_tiny_v2.json"
CORE_SHA = "86c51be6d15f8b1c8bce463a4fb649c4974a339b5bebb20b7592669075620654"
NOTES = ("WHOLE_PAIR_KERNEL_FIXED_PILOT_PROTOCOL_V1.md",
         "WHOLE_PAIR_KERNEL_STATIC_REVIEW_V1.md",
         "WHOLE_PAIR_KERNEL_TINY_V2_PREEXECUTION.md")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(error, reference):
    return float(np.linalg.norm(error)/max(np.linalg.norm(reference), 1e-300))


def tensor(p, lam):
    p, lam = np.asarray(p, dtype=np.float64), np.asarray(lam, dtype=np.float64)
    return np.einsum('ir,jr,kr,r->ijk', p,p,p,lam)


def star_tensor(a):
    a = np.asarray(a, dtype=np.float64)
    eye = np.eye(len(a))
    return (np.einsum('ij,ki->ijk',eye,a)+np.einsum('ik,ji->ijk',eye,a)
            +np.einsum('jk,ij->ijk',eye,a))/3


def distinct_features(p,q):
    n = len(p)
    ii,jj,kk = np.indices((n,n,n))
    distinct = (ii!=jj)&(ii!=kk)&(jj!=kk)
    p,q = p.astype(np.float64),q.astype(np.float64)
    return (np.einsum('ir,jr,kr->ijkr',p,p,p)
            -np.einsum('ir,jr,kr->ijkr',q,q,q))[distinct], distinct


def fixtures():
    rng = np.random.default_rng(20260907)
    n,a = 8,6
    for name in ('dense_signed','correlated_signed'):
        p = np.zeros((n,32),dtype=np.float32)
        weights = np.zeros(32,dtype=np.float32)
        shared = rng.normal(size=(a,1)) / np.sqrt(a)
        plus,minus = [],[]
        for cohort in range(8):
            j = 4*cohort
            scale = rng.uniform(.6,1.4,size=(1,2))
            first = rng.normal(size=(a,2))/np.sqrt(a)
            if name=='correlated_signed':
                first = .4*first+.7*shared
                second = first+.08*rng.normal(size=(a,2))
            else:
                second = rng.normal(size=(a,2))/np.sqrt(a)
            p[:a,j:j+2] = first*scale
            p[:a,j+2:j+4] = second*scale
            w = rng.uniform(.5,1.5,size=2)/(scale.ravel()**3)
            weights[j:j+2] = w
            weights[j+2:j+4] = -w
            if cohort<7:
                plus.extend((j,j+1));minus.extend((j+2,j+3))
        # Untouched newborn follows the parent's shared-offdiagonal format.
        selected = np.array([0,3],dtype=np.int64)
        bp = np.zeros((n,2),dtype=np.float32)
        bp[:a] = rng.normal(size=(a,2))*.1
        bp[selected,np.arange(2)] = 0
        t = np.array([.2,.4],dtype=np.float32)
        s = np.array([-.3,-.1],dtype=np.float32)
        p[:,28:30] = bp;p[:,30:32] = bp
        p[selected,28+np.arange(2)] = t
        p[selected,30+np.arange(2)] = s
        star = (rng.normal(size=(n,n))*.05).astype(np.float32)
        diagonal = np.diag(star).copy()
        star[a:,:]=0;star[:,a:]=0
        np.fill_diagonal(star,diagonal)
        query = rng.normal(size=(n,5))/np.sqrt(n)
        active=np.arange(a,dtype=np.int64)
        if name=='correlated_signed':
            # A fixed relabeling and score-like active order, not new data.
            order=np.array([6,2,4,0,7,1,5,3])
            inverse=np.argsort(order)
            p=p[order];star=star[np.ix_(order,order)];query=query[order]
            bp=bp[order];selected=inverse[selected]
            active=inverse[np.array([4,1,5,0,3,2])]
        yield name,active,p,weights,star,(28,selected,bp,t,s),query,np.array(plus),np.array(minus)


def full_fixture(args):
    name,active0,p0,w0,star0,nb0,query,pi,qi = args
    n = len(p0); a=len(active0); keep = 6
    cold=np.array([i for i in range(n) if i not in active0],dtype=np.int64)
    assert len(np.unique(active0))==a and np.count_nonzero(p0[cold])==0
    captured = {}; originals = {}
    with flops.BudgetContext(flop_budget=2**29, quiet=True) as ctx:
        for key in ('_gpk_norms','_gpk_kernel','_gpk_solve','_gpk_plane'):
            original = getattr(core,key); originals[key]=original
            def wrap(*args, _key=key, _function=original, **kwargs):
                before = int(ctx.flops_used)
                out = _function(*args,**kwargs)
                captured.setdefault(_key,[]).append((out,int(ctx.flops_used)-before))
                return out
            setattr(core,key,wrap)
        try:
            p = fnp.asarray(p0,dtype=fnp.float32)
            w = fnp.asarray(w0,dtype=fnp.float32)
            star = fnp.asarray(star0,dtype=fnp.float32)
            nb = (nb0[0],)+tuple(fnp.asarray(x) for x in nb0[1:])
            active = fnp.asarray(active0,dtype=fnp.int64)
            got = core._gpk_compress(p,w,star,nb,[2]*8,active,
                                     lambda left,right: left@right,keep=keep)
            total = int(ctx.flops_used)
        finally:
            for key,original in originals.items():setattr(core,key,original)
    newp,neww,newstar,newborn = got
    newp,neww,newstar = (np.asarray(x) for x in (newp,neww,newstar))
    norms = np.asarray(captured['_gpk_norms'][0][0][0])
    kernel32 = np.asarray(captured['_gpk_kernel'][0][0])
    gram64 = np.asarray(captured['_gpk_kernel'][1][0])
    selected,stored = (np.asarray(x) for x in captured['_gpk_solve'][0][0])
    phi,distinct = distinct_features(p0[:,pi],p0[:,qi])
    norms_ref = np.sum(phi*phi,axis=0)
    energy = w0[pi].astype(np.float64)**2*norms_ref
    selected_ref = np.argsort(-energy,kind='stable')[:keep]
    scale = np.sqrt(norms_ref[selected_ref])
    cross_ref = phi[:,selected_ref].T@phi
    gram_ref = cross_ref[:,selected_ref]
    rhs_ref = cross_ref@w0[pi].astype(np.float64)/scale
    g = gram_ref/(scale[:,None]*scale[None,:])+1e-3*np.eye(keep)
    c_ref = np.linalg.solve(g,rhs_ref)
    physical_ref = c_ref/scale
    rhs32 = kernel32@w0[pi].astype(np.float64)/np.sqrt(norms[selected])
    old_tensor = tensor(p0,w0)+star_tensor(star0)
    new_tensor = tensor(newp,neww)+star_tensor(newstar)
    residual = old_tensor-new_tensor
    projected = np.where(distinct,residual,0)
    residual_query = np.einsum('ijk,iu,jv,kw->uvw',residual,query,query,query)
    projected_query = np.einsum('ijk,iu,jv,kw->uvw',projected,query,query,query)
    old_query = np.einsum('ijk,iu,jv,kw->uvw',old_tensor,query,query,query)
    reference_indices = np.concatenate((pi[selected_ref],qi[selected_ref]))
    objective = lambda v: float(np.sum((phi[:,selected_ref]@v-phi@w0[pi])**2)
                                +1e-3*np.sum((v*scale)**2))
    checks = dict(norms_relative=relative(norms-norms_ref,norms_ref),
                  gram64_relative=relative(gram64-gram_ref,gram_ref),
                  cross32_relative=relative(kernel32-cross_ref,cross_ref),
                  rhs32_relative=relative(rhs32-rhs_ref,rhs_ref),
                  physical_coefficient_relative=relative(stored-physical_ref,physical_ref),
                  repeated_tensor_relative=relative(residual[~distinct],old_tensor),
                  query_repeated_relative=relative(residual_query-projected_query,old_query),
                  ridge_objective_actual=objective(stored.astype(np.float64)),
                  ridge_objective_reference=objective(physical_ref))
    gates = dict(selection=bool(np.array_equal(selected,selected_ref)),
                 norms=checks['norms_relative']<=1e-10,
                 gram64=checks['gram64_relative']<=1e-10,
                 cross32=checks['cross32_relative']<=1e-4,
                 coefficients=checks['physical_coefficient_relative']<=1e-3,
                 repeated=checks['repeated_tensor_relative']<=3e-6,
                 query=checks['query_repeated_relative']<=1e-5,
                 stored_physical_factors=bool(np.array_equal(newp[:,:2*keep],p0[:,reference_indices])),
                 paired_weights=bool(np.array_equal(neww[:keep],-neww[keep:2*keep])),
                 newborn_p=bool(np.array_equal(newp[:,2*keep:],p0[:,28:])),
                 newborn_weights=bool(np.array_equal(neww[2*keep:],w0[28:])),
                 newborn_offset=newborn[0]==2*keep,
                 newborn_fields=all(newborn[i] is nb[i] for i in range(1,5)),
                 cold_star=bool(np.array_equal(newstar[cold,:],star0[cold,:])
                                and np.array_equal(newstar[:,cold],star0[:,cold])),
                 finite=bool(np.isfinite(newp).all() and np.isfinite(neww).all()
                             and np.isfinite(newstar).all()))
    return dict(fixture=name,active_indices=active0.tolist(),passed=all(gates.values()),gates=gates,checks=checks,
                total_native_flops=total,
                inclusive_function_flops={key:[x[1] for x in value] for key,value in captured.items()},
                selected_indices=selected.tolist(),reference_selected_indices=selected_ref.tolist(),
                normalized_reference_gram_condition=float(np.linalg.cond(g)),
                input_sha256=hashlib.sha256(p0.tobytes()+w0.tobytes()+star0.tobytes()).hexdigest())


def reference_coefficients(p,q,weights,selected):
    phi,_=distinct_features(p,q)
    scale=np.sqrt(np.sum(phi[:,selected]**2,axis=0))
    cross=phi[:,selected].T@phi
    gram=cross[:,selected]/(scale[:,None]*scale[None,:])
    rhs=cross@weights.astype(np.float64)/scale
    return np.linalg.solve(gram+1e-3*np.eye(len(selected)),rhs)/scale


def boundaries():
    # Predetermined coordinate triples make exact energy ties/zero structure.
    p0 = np.array([[1,1,1],[1,1,0],[1,1,1],[0,0,1],[0,0,0],[0,0,0]],dtype=np.float32)
    q0 = np.zeros_like(p0)
    records=[]
    with flops.BudgetContext(flop_budget=2**27,quiet=True) as ctx:
        p,q = fnp.asarray(p0),fnp.asarray(q0)
        w = fnp.asarray(np.ones(3,dtype=np.float32))
        sel,coeff = core._gpk_solve(p,q,w,2)
        selected=np.asarray(sel)
        cref=reference_coefficients(p0,q0,np.ones(3,dtype=np.float32),selected)
        cerr=relative(np.asarray(coeff)-cref,cref)
        records.append(dict(case='ties_and_duplicate_dictionary',
                            passed=bool(np.array_equal(selected,[0,1])
                                        and np.isfinite(np.asarray(coeff)).all()
                                        and cerr<=1e-3),
                            physical_coefficient_relative=cerr))
        zp=fnp.asarray(np.array([[1],[0],[0],[0],[0],[0]],dtype=np.float32))
        zq=fnp.zeros_like(zp)
        rejected=False
        try:core._gpk_solve(zp,zq,fnp.ones(1,dtype=fnp.float32),1)
        except ValueError as exc:rejected='Insufficient nonzero' in str(exc)
        records.append(dict(case='zero_distinct_rejected',passed=rejected))
        hp=np.array([[1],[.5],[-.75],[.25],[.625],[-.375]],dtype=np.float32)
        hq=hp+np.array([[1],[-1],[2],[0],[-2],[1]],dtype=np.float32)*2**-10
        hf=core._gpk_features(fnp.asarray(hp),fnp.asarray(hq))
        hn,_=core._gpk_norms(hf)
        hphi,_=distinct_features(hp,hq)
        href=np.sum(hphi*hphi,axis=0)
        err=relative(np.asarray(hn)-href,href)
        hs,hc=core._gpk_solve(fnp.asarray(hp),fnp.asarray(hq),fnp.ones(1,dtype=fnp.float32),1)
        hcref=reference_coefficients(hp,hq,np.ones(1,dtype=np.float32),np.asarray(hs))
        hcerr=relative(np.asarray(hc)-hcref,hcref)
        records.append(dict(case='near_cancellation_original_cubes',
                            passed=bool(err<=1e-10 and hcerr<=1e-3),
                            norm_relative=err,physical_coefficient_relative=hcerr))
        total=int(ctx.flops_used)
    return dict(passed=all(r['passed'] for r in records),tests=records,total_native_flops=total)


def main():
    if OUT.exists():raise FileExistsError(OUT)
    paths=[Path(__file__),HERE/'whole_pair_kernel_core_v1.py']+[HERE/x for x in NOTES]
    before={p.name:digest(p) for p in paths}
    result=dict(passed=False,tests=[],failure=None,seals_before=before)
    start=time.time()
    try:
        assert before['whole_pair_kernel_core_v1.py']==CORE_SHA
        env={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
        nice=os.getpriority(os.PRIO_PROCESS,0)
        versions={k:importlib.metadata.version(k) for k in ('flopscope','numpy')}
        result.update(blas_env=env,nice=nice,versions=versions,interpreter=sys.executable)
        assert set(env.values())=={'1'} and nice>=10
        assert versions['flopscope']=='0.12.0'
        for args in fixtures():
            record=full_fixture(args);result['tests'].append(record)
            print(json.dumps(record),flush=True)
        result['boundaries']=boundaries()
        result['passed']=all(r['passed'] for r in result['tests']) and result['boundaries']['passed']
    except Exception as exc:result['failure']=type(exc).__name__+': '+str(exc)
    after={p.name:digest(p) for p in paths}
    result.update(seals_after=after,seals_unchanged=before==after,
                  wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  qualification='Fixed synthetic original-cube implementation/API certificate only. No P2 state, targets, raw/adjusted score, native candidate bill or production runtime.')
    result['passed']=bool(result['passed'] and result['failure'] is None and before==after)
    with OUT.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
