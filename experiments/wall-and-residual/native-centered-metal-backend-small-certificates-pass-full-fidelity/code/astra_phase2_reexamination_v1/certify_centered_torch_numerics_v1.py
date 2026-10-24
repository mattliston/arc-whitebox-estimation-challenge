"""Target-free fixed scalar/pair values and reverse-gradient certificates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from centered_relu_marginals_v1 import marginals as numpy_marginals
from centered_pair_numerics_v1 import nonlinear as numpy_nonlinear
from centered_torch_numerics_v1 import marginals, nonlinear, normal_cdf
from full_carrier_autodiff_v1 import raw_marginals, nonlinear as raw_nonlinear

HERE = Path(__file__).resolve().parent


def host(t):
    return t.detach().cpu().to(torch.float64).numpy()


def scalar_outputs(fn, x, sigma_fixed):
    a, log_s, skew, kurt = x.unbind(1)
    s = log_s.exp()
    mean, aux = fn(a*s, s*s, skew*s**3, kurt*s**4)
    return (mean/sigma_fixed, aux['post_v']/sigma_fixed**2,
            aux['post_k3']/sigma_fixed**3, aux['post_k4']/sigma_fixed**4)


def scalar_check(device, dtype):
    cells = [(a,s,k3,k4) for a in (-12.,-8.,-4.,-2.,-.5,0.,.5,2.,4.,8.,12.)
             for s in (.3,1.7) for k3,k4 in ((0.,0.),(.02,.03),(-.015,-.02))]
    data = np.asarray(cells)
    al, si, sk, ku = data.T
    nm, na = numpy_marginals(al*si, si**2, sk*si**3, ku*si**4)
    expected = (nm/si, na['post_v']/si**2, na['post_k3']/si**3, na['post_k4']/si**4)
    x_np = np.stack((al,np.log(si),sk,ku),axis=1)
    x = torch.tensor(x_np,device=device,dtype=dtype,requires_grad=True)
    sf = torch.tensor(si,device=device,dtype=dtype)
    values = scalar_outputs(marginals,x,sf)
    xr = torch.tensor(x_np,dtype=torch.float64,requires_grad=True)
    ref = scalar_outputs(raw_marginals,xr,torch.tensor(si,dtype=torch.float64))
    value_errors, grad_errors, zero_errors = [], [], []
    for j in range(4):
        value_errors.append(float(np.max(np.abs(host(values[j])-expected[j]))))
        grad = host(torch.autograd.grad(values[j].sum(),x,retain_graph=True)[0])
        refgrad = host(torch.autograd.grad(ref[j].sum(),xr,retain_graph=True)[0])
        err = np.abs(grad-refgrad)
        grad_errors.append(float(np.max(err)))
        zero_errors.append(float(np.max(err[al==0])))
    vtol, gtol = ((1e-10,1e-9) if dtype == torch.float64 else (2e-5,2e-4))
    return dict(dtype=str(dtype), cells=len(cells), forward_max_by_order=value_errors,
                derivative_max_by_order=grad_errors, zero_mean_derivative_max_by_order=zero_errors,
                passed=max(value_errors)<=vtol and max(grad_errors)<=gtol)


def pair_check(device,dtype):
    rng = np.random.Generator(np.random.PCG64DXSM(930092))
    rows = []
    for n in (11,17):
        for asymmetry in (0.,1e-6):
            X = rng.normal(size=(n,n))
            C = X@X.T/n+.4*np.eye(n)
            N = rng.normal(size=(n,n)); C += asymmetry*(N-N.T)
            v = np.diag(C).copy(); s = np.sqrt(v)
            a = rng.uniform(-3,3,n); a[0] = 0
            mu = a*s
            K = rng.normal(size=(n,n))*.002*(v[:,None]*s[None,:])
            radial = np.asarray(.006)
            inputs = (mu,C,K,radial)
            nr = numpy_nonlinear(mu,C,K,radial*v*v,radial*(np.outer(v,v)+2*C*C)/3,'covariance')
            ts = tuple(torch.tensor(z,device=device,dtype=dtype,requires_grad=True) for z in inputs)
            rs = tuple(torch.tensor(z,dtype=torch.float64,requires_grad=True) for z in inputs)
            new, ref = nonlinear(*ts), raw_nonlinear(*rs)
            scales = (s,np.outer(s,s),v[:,None]*s[None,:],np.asarray(1.))
            forward = [float(np.max(np.abs(host(new[j])-nr[j])/scales[j])) for j in range(4)]
            input_scales = (s,np.outer(s,s),v[:,None]*s[None,:],np.asarray(1.))
            gradient_relative = []
            for j in range(4):
                cot = rng.normal(size=np.shape(nr[j]))/scales[j]
                cot = cot/max(float(np.linalg.norm(cot)),1e-30)
                obj = (new[j]*torch.tensor(cot,device=device,dtype=dtype)).sum()
                refobj = (ref[j]*torch.tensor(cot,dtype=torch.float64)).sum()
                ng = torch.autograd.grad(obj,ts,retain_graph=True)
                rg = torch.autograd.grad(refobj,rs,retain_graph=True)
                gn = np.concatenate([(host(z)*sc).ravel() for z,sc in zip(ng,input_scales)])
                gr = np.concatenate([(host(z)*sc).ravel() for z,sc in zip(rg,input_scales)])
                gradient_relative.append(float(np.linalg.norm(gn-gr)/max(np.linalg.norm(gr),1e-30)))
            vt,gt = ((1e-10,1e-9) if dtype==torch.float64 else (2e-5,2e-4))
            rows.append(dict(width=n,input_asymmetry=asymmetry,dtype=str(dtype),
                             forward_normalized_max=forward,gradient_relative=gradient_relative,
                             passed=max(forward)<=vt and max(gradient_relative)<=gt))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device',choices=('cpu','mps'),default='cpu')
    args = ap.parse_args()
    out = HERE/f'centered_torch_numerics_{args.device}_small_v1.json'
    assert not out.exists()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0') == '0'
    assert 'threadpoolctl' not in __import__('sys').modules
    parent = json.loads((HERE/'centered_pair_numerics_small_v1.json').read_text())
    assert parent['passed']
    torch.set_num_threads(3)
    start = time.time()
    result = dict(device=args.device,torch_version=torch.__version__,target_access=False,
                  fallback_enabled=False,source_sha256=hashlib.sha256((HERE/'centered_torch_numerics_v1.py').read_bytes()).hexdigest())
    try:
        # This is intentionally a direct backend call: unsupported erfc fails.
        probe = torch.tensor([-12.,0.,12.],device=args.device,requires_grad=True)
        cdf = normal_cdf(probe)
        cdf.sum().backward()
        result['erfc_probe_values'] = host(cdf).tolist()
        result['erfc_probe_derivatives'] = host(probe.grad).tolist()
        result['scalar'] = []
        result['pair'] = []
        for dtype in ((torch.float64,torch.float32) if args.device=='cpu' else (torch.float32,)):
            row = scalar_check(args.device,dtype)
            result['scalar'].append(row)
            print(json.dumps(dict(stage='scalar',device=args.device,**row)),flush=True)
            rows = pair_check(args.device,dtype)
            result['pair'].extend(rows)
            for row in rows:
                print(json.dumps(dict(stage='pair',device=args.device,**row)),flush=True)
        if args.device=='mps':
            torch.mps.synchronize()
        result['passed'] = all(r['passed'] for r in result['scalar']+result['pair'])
    except Exception as exc:
        result['passed'] = False
        result['exception'] = dict(type=type(exc).__name__,message=str(exc))
    result.update(wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification='Fixed small-map numerical and reverse-gradient equivalence only. No full-depth parity, training, deployed candidate, raw/adjusted or score claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('scalar','pair')},indent=2),flush=True)
    assert result['passed'],result


if __name__=='__main__':
    main()
