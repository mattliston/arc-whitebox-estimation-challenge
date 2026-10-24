"""Add target-free prevariance-domain auditing to the frozen wide checks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','mps'))
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.device is None: parser.print_help(); return
    assert args.out is not None and not args.out.exists()
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert 'threadpoolctl' not in sys.modules
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0'
    import qualify_model as q
    pins={str(p):q.sha(p) for p in (HERE/'model.py',HERE/'qualify_model.py',HERE/'PROTOCOL.md',Path(__file__))}
    args.out.mkdir()
    started=time.monotonic()
    result=dict(passed=False,device=args.device,sources=pins,truth_accessed=False,
                started_utc=datetime.now(timezone.utc).isoformat())
    try:
        import numpy as np
        import torch
        import model
        torch.set_num_threads(3)
        original=model.env['nonlinear']
        domain=[]
        def audited(mu,cov,k21,radial):
            v=torch.diagonal(cov).detach()
            domain.append(dict(minimum_variance=float(v.min().cpu()),
                               below_floor_count=int((v<1e-10).sum().cpu())))
            return original(mu,cov,k21,radial)
        model.env['nonlinear']=audited
        result['checks']=q.wide(np,torch,model,args.device,args.out)
        result['variance_domain']=dict(calls=len(domain),
            minimum_variance=min(x['minimum_variance'] for x in domain),
            below_floor_count=sum(x['below_floor_count'] for x in domain))
        assert result['variance_domain']['below_floor_count']==0, 'Centered/raw variance-domain difference'
        assert all(q.sha(p)==v for p,v in pins.items())
        result.update(passed=True,**q.resources(torch))
    except BaseException:
        result['error']=traceback.format_exc()
    result.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started)
    q.write(args.out/'result.json',result)
    print(json.dumps(result),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
