"""Tiny compatibility diagnostic; no card/target calls."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

HERE=Path(__file__).resolve().parent


def main():
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('30s cap')))
    signal.alarm(30);start=time.monotonic()
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise RuntimeError('nice10 required')
    out=HERE/'ad_scalar_probe_v1';out.mkdir(exist_ok=False)
    result={'diagnostic_only':True,'model_calls':0,'truth_accessed':False,'cases':[],
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(256*2**20/torch.mps.recommended_max_memory())
    for device in ('cpu','mps'):
        x=torch.tensor([.13,.25,.31,.42,.57,.68,.72,.83],dtype=torch.float32,device=device)
        dx=torch.tensor([.11,-.21,.12,.15,-.07,.03,.08,-.11],dtype=torch.float32,device=device)
        den=torch.tensor(math.sqrt(8),dtype=torch.float32,device=device)
        rec=torch.tensor(1/math.sqrt(8),dtype=torch.float32,device=device)
        funcs={'python_float_divide':lambda z:z.sum()/math.sqrt(8),
               'typed_tensor_divide':lambda z:z.sum()/den,
               'typed_tensor_reciprocal_multiply':lambda z:z.sum()*rec,
               'integer_divide':lambda z:z.sum()/64}
        for name,fn in funcs.items():
            item={'device':device,'case':name}
            try:
                normal=fn(x)
                bx=x.detach().clone().requires_grad_(True)
                loss=fn(bx);loss.backward()
                rev=float((bx.grad*dx).sum().detach().cpu())
                with torch.autograd.forward_ad.dual_level():
                    with torch.no_grad():y=fn(torch.autograd.forward_ad.make_dual(x,dx))
                    primal,tan=torch.autograd.forward_ad.unpack_dual(y)
                    item.update(primal=float(primal.cpu()),ordinary_primal=float(normal.cpu()),
                                primal_byte_exact=bool(torch.equal(primal,normal)),
                                forward_ad=float(tan.cpu()),reverse_dot=rev,
                                absolute_identity_error=abs(float(tan.cpu())-rev),supported=True)
            except Exception as e:item.update(supported=False,error=f'{type(e).__name__}: {e}')
            result['cases'].append(item)
        del x,dx,den,rec,funcs
    torch.mps.synchronize()
    result.update(elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  driver_bytes=torch.mps.driver_allocated_memory())
    result['resource_passed']=result['elapsed_s']<30 and result['peak_rss_bytes']<1e9 and result['driver_bytes']<256*2**20
    (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (out/'completion.json').write_text(json.dumps(dict(completed=True,resource_passed=result['resource_passed'],
        result_sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest()))+'\n')
    print(json.dumps(result,indent=2));signal.alarm(0)


if __name__=='__main__':main()
