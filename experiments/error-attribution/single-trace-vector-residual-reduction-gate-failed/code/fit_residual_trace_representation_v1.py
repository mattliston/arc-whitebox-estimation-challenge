"""Oracle source-slice trace-vector fit; no next-plane targets in the fit."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
from residual_one_high_oracle_v1 import distinct_plane


def slices(b,C,U):
    return np.stack([(b@u)*C+np.outer(b,C@u)+np.outer(C@u,b) for u in U.T])


def normal(C,U):
    V=C@U;A=C@V
    return (2*np.sum(V*V)*np.eye(len(C))+np.sum(C*C)*(U@U.T)
            +2*(U@A.T+A@U.T+V@V.T))


def rhs(M,C,U):
    return sum(u*np.sum(C*m)+2*m@(C@u) for u,m in zip(U.T,M))


def certificate():
    rng=np.random.default_rng(291765);n=9;q=4
    A=rng.normal(size=(n,n));C=A@A.T/n+np.eye(n)*.1
    U=np.linalg.qr(rng.normal(size=(n,q)))[0]
    D=np.stack([slices(e,C,U).ravel() for e in np.eye(n)],axis=1)
    M=rng.normal(size=(q,n,n));M=(M+M.transpose(0,2,1))/2
    N=normal(C,U);b=np.linalg.solve(N,rhs(M,C,U))
    ref=np.linalg.lstsq(D,M.ravel(),rcond=None)[0]
    errors=dict(normal=float(np.max(np.abs(N-D.T@D))),rhs=float(np.max(np.abs(rhs(M,C,U)-D.T@M.ravel()))),
                solution=float(np.max(np.abs(b-ref))))
    assert errors['normal']<1e-12 and errors['rhs']<1e-12 and errors['solution']<1e-12,errors
    return dict(passed=True,errors=errors)


def main():
    HERE=Path(__file__).resolve().parent
    OUT=HERE/'residual_trace_representation_r35_l12_v1.json';ART=OUT.with_suffix('.npz')
    assert not OUT.exists() and not ART.exists()
    assert json.loads((HERE/'residual_one_high_oracle_r35_l12_v1.json').read_text())['opportunity_gate']
    small=certificate();start=time.time()
    fpath=HERE/'residual_one_high_inputs_r35_l12_v1.npz';f=np.load(fpath)
    U=f['U'];C=f['post_covariance'];W=f['next_weight'];indices=f['indices']
    targets={};corrections={};art={};fit_stats={}
    with threadpool_limits(limits=4):
        N=normal(C,U)
        for name in ('A','B'):
            t=np.load(HERE/f'residual_one_high_truth_r35_l12_{name}_v1.npz')
            M=t['slices']-f['model_slices']
            R=rhs(M,C,U);b=np.linalg.solve(N,R)
            residual=float(np.linalg.norm(N@b-R)/np.linalg.norm(R));assert residual<1e-10
            fitted=slices(b,C,U)
            correction=distinct_plane(fitted,U,W,indices)
            targets[name]=t['next_K21']-f['model_next_K21'];corrections[name]=correction
            fit_stats[name]=dict(normal_residual=residual,source_slice_relative_error=float(np.linalg.norm(fitted-M)/np.linalg.norm(M)),
                                 vector_norm=float(np.linalg.norm(b)))
            art.update({f'{name}_b':b,f'{name}_correction':correction})
        def metrics(weight):
            EA=targets['A']*weight;EB=targets['B']*weight
            FA=corrections['A']*weight;FB=corrections['B']*weight
            s=float(np.mean(EA*EB));c=float(np.mean(FA*FB));a=float(.5*np.mean(EA*FB+EB*FA))
            assert s>0 and c>0
            return dict(signal_error_ratio=(s-2*a+c)/s,signal_cosine=a/np.sqrt(s*c),
                        signal_correction_relative_norm=np.sqrt(c/s),
                        A_target_B_correction_ratio=float(np.sum((EA-FB)**2)/np.sum(EA*EA)),
                        B_target_A_correction_ratio=float(np.sum((EB-FA)**2)/np.sum(EB*EB)))
        result=dict(unweighted=metrics(1.),leading_consumer_weighted=metrics(f['consumer_weight']))
    v=result['leading_consumer_weighted']
    gate=v['signal_error_ratio']<=.7 and v['signal_cosine']>=.6 and v['A_target_B_correction_ratio']<1 and v['B_target_A_correction_ratio']<1
    np.savez(ART,**art)
    record=dict(rank=35,layer=12,q=4,certificate=small,fit_stats=fit_stats,results=result,representation_gate=bool(gate),
                frozen_input_sha256=hashlib.sha256(fpath.read_bytes()).hexdigest(),wall_s=time.time()-start,
                peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                qualification='Oracle b fitted only to source-slice residuals, separately per independent stream. Sym(b,C) then one-high projection and coordinate-repeat removal. No next-plane target fit, deployable source, score or confidence bound.')
    OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
