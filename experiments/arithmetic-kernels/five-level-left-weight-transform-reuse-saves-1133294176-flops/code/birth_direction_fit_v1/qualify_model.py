"""Explicit target-free small and full-width implementation qualification."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def resources(torch):
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    driver = int(torch.mps.driver_allocated_memory()) if torch.backends.mps.is_available() else 0
    assert rss < 12e9 and driver < 16e9, (rss, driver)
    return dict(peak_rss_bytes=rss, mps_driver_bytes=driver)


def tiny(np, torch, model):
    rng = np.random.default_rng(9309081)
    n = 7
    z = rng.normal(size=(n,n))
    C = z@z.T/n + np.eye(n)
    sig = np.sqrt(np.diag(C)); alpha = np.linspace(-1.9,2.1,n)
    pdf = np.exp(-alpha**2/2)/np.sqrt(2*np.pi)
    gate = np.asarray([.5*__import__('math').erfc(-x/np.sqrt(2)) for x in alpha])
    K = rng.normal(size=(n,n))*.02
    B = gate[:,None]*C; np.fill_diagonal(B,0)
    alive = np.array([1,1,0,1,1,1,0.])
    births = np.array([1,0,1,1,0,1,0.])
    t = np.array([2.5,-.3,1.4,.01,.2,-1.2,.8])
    s = t-np.linspace(.15,1.1,n); a = np.linspace(.1,.7,n)
    c = np.array([.2,-.3,.1,.15],np.float32)
    aux = dict(sigma=sig,alpha=alpha,pdf=pdf,gate=gate)
    T = lambda x:torch.tensor(x,dtype=torch.float64)
    taux = {k:T(v) for k,v in aux.items()}
    basis = model.residualized_basis(T(B),T(C),T(K),taux)
    # Independently form the four unprojected NumPy directions.
    q,d = alpha/sig,pdf/sig
    raw = [.5*d[:,None]*K,
           -.5*gate[:,None]*K.T*q[None,:],
           -.5*d[:,None]*C**2*q[None,:],
           C*((np.diag(K)/sig**3)*(alpha**2-1)*pdf/6)[:,None]]
    for v in raw:
        np.fill_diagonal(v,0)
        v -= B*np.sum(B*v,axis=0)[None,:]/np.maximum(np.sum(B*B,axis=0),1e-20)[None,:]
    basis_error = max(float(np.max(np.abs(v.detach().numpy()-r))) for v,r in zip(basis,raw))
    assert basis_error < 1e-14
    new, change = model.birth_action(T(B),T(C),T(K),taux,T(c),T(t),T(s),T(a),T(alive),T(births))
    old = (B*alive[:,None]).astype(np.float32)
    tt = (t*alive).astype(np.float32).astype(float)
    ss = (s*alive).astype(np.float32).astype(float)
    aa = a.astype(np.float32).astype(float)*births

    def tensor(base):
        p,q = base.astype(float).copy(),base.astype(float).copy()
        np.fill_diagonal(p,tt); np.fill_diagonal(q,ss)
        return sum(aa[j]*(np.einsum('i,j,k->ijk',p[:,j],p[:,j],p[:,j])
                         -np.einsum('i,j,k->ijk',q[:,j],q[:,j],q[:,j])) for j in range(n))

    delta_tensor = tensor(new.detach().numpy())-tensor(old)
    literal = np.einsum('iij->ij',delta_tensor)
    plane_error = float(np.max(np.abs(literal-change.detach().numpy())))
    assert plane_error < 1e-13, plane_error
    zero_new, zero_change = model.birth_action(T(B),T(C),T(K),taux,torch.zeros(4,dtype=torch.float64),T(t),T(s),T(a),T(alive),T(births))
    assert zero_new.detach().numpy().tobytes()==old.tobytes()
    assert torch.count_nonzero(zero_change)==0
    # Exact repeated restoration leaves precisely the all-distinct change.
    A = -3*literal.T+2*np.diag(np.diag(literal))
    star = np.empty((n,n,n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                star[i,j,k]=(A[i,j]*(j==k)+A[j,i]*(i==k)+A[k,i]*(i==j))/3
    restored = delta_tensor+star
    mask = np.fromfunction(lambda i,j,k:(i==j)|(i==k)|(j==k),(n,n,n))
    assert np.max(np.abs(restored[mask]))<1e-13
    assert np.max(np.abs(restored[~mask]-delta_tensor[~mask]))==0
    # Smooth FP64 algebra: projection derivatives include B and K changes.
    bt,kt = T(B).requires_grad_(),T(K).requires_grad_()
    cot = T(rng.normal(size=(n,n)))
    value = sum((x*cot).sum()*(j+1) for j,x in enumerate(model.residualized_basis(bt,T(C),kt,taux)))
    value.backward()
    db,dk = T(rng.normal(size=(n,n))),T(rng.normal(size=(n,n)))
    db = model.diagonal(db,torch.zeros(n,dtype=torch.float64))
    analytic = float((bt.grad*db).sum()+(kt.grad*dk).sum())
    eps=1e-5
    def functional(sign):
        return float(sum((x*cot).sum()*(j+1) for j,x in enumerate(model.residualized_basis(T(B)+sign*eps*db,T(C),T(K)+sign*eps*dk,taux))))
    finite = (functional(1)-functional(-1))/(2*eps)
    gradient_relative = abs(finite-analytic)/max(abs(analytic),1e-12)
    assert gradient_relative<1e-6
    # Below the projection floor it is residualized, not exactly orthogonal.
    tiny_B=T(B*1e-11)
    below=model.residualized_basis(tiny_B,T(C),T(K),taux)
    for j,v in enumerate(below):
        raw_j=model.diagonal((.5*T(d)[:,None]*T(K),
            -.5*T(gate)[:,None]*T(K).T*T(q)[None,:],
            -.5*T(d)[:,None]*T(C)**2*T(q)[None,:],
            T(C)*(T(np.diag(K)/sig**3)*(T(alpha)**2-1)*T(pdf)/6)[:,None])[j],
            torch.zeros(n,dtype=torch.float64))
        expected=(tiny_B*raw_j).sum(0)*(1-(tiny_B*tiny_B).sum(0)/1e-20)
        assert torch.max(torch.abs((tiny_B*v).sum(0)-expected))<1e-20
    return dict(basis_max_error=basis_error,stored_plane_max_error=plane_error,
                null_bytes_exact=True,repeated_restoration=True,
                projection_directional_gradient_relative=gradient_relative,
                below_floor_projection_checked=True)


def wide(np, torch, model, device, out):
    entries=[]
    native = PARENT/'unused_materializations_v1/native'
    receipt=json.loads((native/'result.json').read_text())
    assert sha(native/'result.json')=='a9ccecaa3c0ca79f070af3c0961a6a952f08f4d9712e82a3d6af4d9a0d1d0951'
    for rank,call in ((35,0),(100,1)):
        fixed = (.1*np.sin(np.arange(60,dtype=float)+.3)).reshape(15,4).astype(np.float32)
        coeff = torch.tensor(model.deployment_coefficients(fixed),device=device)
        for name,values in (('zero',torch.zeros_like(coeff)),('nonzero',coeff)):
            with torch.no_grad():
                card=model.BirthDirectionCard(rank,coefficients=values,device=device)
                prediction,trace=card.run(capture=True)
                prediction=prediction.clamp(min=0).cpu().float().numpy()
                assert np.isfinite(prediction).all()
            path=out/f'{device}_{rank}_{name}.npy'
            with path.open('xb') as f: np.save(f,prediction)
            item=dict(rank=rank,action=name,prediction_sha256=sha(path),trace=trace)
            if name=='zero':
                filename=f'call_{call:02d}_rank{rank}_output.npy'
                oldpath=native/filename
                assert sha(oldpath)==receipt['artifacts'][filename]['sha256']
                old=np.load(oldpath)
                error=float(np.max(np.abs(prediction.astype(float)-old.astype(float))))
                item['current_native_null_max_difference']=error
                assert error<=1e-5,error
            del card,prediction
            gc.collect()
            if device=='mps': torch.mps.empty_cache()
            item.update(resources(torch)); entries.append(item)
            print(json.dumps(item),flush=True)
        theta=torch.tensor(fixed,device=device,requires_grad=True)
        card=model.BirthDirectionCard(rank,theta=theta,device=device)
        prediction,_=card.run(training=True)
        # The cotangent is fixed and target-free. No mean labels are opened.
        cot=np.sin(np.arange(1024,dtype=float)*.37).astype(np.float32)/1024
        loss=(prediction[-1]*torch.tensor(cot,device=device)).sum()
        loss.backward()
        grad=theta.grad.detach().cpu().float().numpy().copy()
        assert np.isfinite(grad).all() and np.linalg.norm(grad)>1e-10
        with (out/f'{device}_{rank}_gradient.npy').open('xb') as f: np.save(f,grad)
        item=dict(rank=rank,gradient_norm=float(np.linalg.norm(grad)),**resources(torch))
        entries.append(item); print(json.dumps(item),flush=True)
        del card,prediction,theta,loss
        gc.collect()
        if device=='mps': torch.mps.empty_cache()
    return entries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('small','cpu','mps'))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.stage is None: parser.print_help(); return
    assert args.out is not None and not args.out.exists()
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert 'threadpoolctl' not in sys.modules
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0'
    sources={str(p):sha(p) for p in (HERE/'model.py',HERE/'PROTOCOL.md',Path(__file__))}
    args.out.mkdir()
    started=time.monotonic()
    result=dict(passed=False,stage=args.stage,started_utc=datetime.now(timezone.utc).isoformat(),sources=sources,truth_accessed=False)
    try:
        import numpy as np
        import torch
        import model
        torch.set_num_threads(3)
        if args.stage=='small': result['checks']=tiny(np,torch,model)
        else: result['checks']=wide(np,torch,model,args.stage,args.out)
        assert all(sha(p)==v for p,v in sources.items())
        result.update(passed=True,**resources(torch))
    except BaseException:
        result['error']=traceback.format_exc()
    result.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started)
    write(args.out/'result.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
