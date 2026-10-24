"""Fixed CPU/MPS CDF and derivative certificate against analytic Gaussian."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import torch
from gaussian_cdf_quadrature_v1 import normal_cdf

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--device',choices=('cpu','mps'),default='cpu')
args = ap.parse_args()
out = HERE/f'gaussian_cdf_quadrature_{args.device}_v1.json'
assert not out.exists()
assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0')=='0'
torch.set_num_threads(3)
x0 = torch.linspace(-16,16,4017,dtype=torch.float64)
ref = torch.erfc(-x0/math.sqrt(2))/2
derivative = torch.exp(-x0*x0/2)/math.sqrt(2*math.pi)
rows = []
start = time.time()
for dtype in ((torch.float64,torch.float32) if args.device=='cpu' else (torch.float32,)):
    x = x0.to(device=args.device,dtype=dtype).requires_grad_(True)
    y = normal_cdf(x)
    grad = torch.autograd.grad(y.sum(),x)[0]
    yc,gc = y.detach().cpu().double(),grad.detach().cpu().double()
    ve = float((yc-ref).abs().max())
    ge = float((gc-derivative).abs().max())
    vt,gt = ((1e-11,1e-10) if dtype==torch.float64 else (2e-7,2e-6))
    rows.append(dict(dtype=str(dtype),max_cdf_error=ve,max_derivative_error=ge,
                     cdf_zero=float(yc[2008]),derivative_zero=float(gc[2008]),
                     passed=ve<=vt and ge<=gt and float(yc[2008])==.5))
result = dict(passed=all(r['passed'] for r in rows),records=rows,points=4017,
              device=args.device,fallback_enabled=False,target_access=False,
              source_sha256=hashlib.sha256((HERE/'gaussian_cdf_quadrature_v1.py').read_bytes()).hexdigest(),
              wall_s=time.time()-start,utc_finished=datetime.now(timezone.utc).isoformat(),
              qualification='Fixed normal CDF numerical value/derivative approximation; no moment map, full-depth, training or score claim.')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
assert result['passed'],result
