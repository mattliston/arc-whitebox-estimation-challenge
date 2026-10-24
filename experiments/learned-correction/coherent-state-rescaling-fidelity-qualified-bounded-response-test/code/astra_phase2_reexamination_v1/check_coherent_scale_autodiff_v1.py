"""Full CPU dynamic secants and zero-point tangent vs independent NumPy."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.autograd import forward_ad
from coherent_central_scale_autodiff_v1 import CoherentScaleCard

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank',type=int,choices=(35,100),required=True)
    args = ap.parse_args()
    out = HERE/f'coherent_scale_autodiff_r{args.rank}_v1.json'
    arrays = out.with_suffix('.npz')
    assert not out.exists() and not arrays.exists()
    ref = np.load(HERE/f'coherent_scale_numpy_fidelity_r{args.rank}_v1.npz')
    npr = json.loads((HERE/f'coherent_scale_numpy_fidelity_r{args.rank}_v1.json').read_text())
    torch.set_num_threads(3)
    card = CoherentScaleCard(args.rank)
    direction = torch.tensor(ref['direction'])
    rel = lambda a,b:float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
    rows, saves = [], {}
    t0 = time.time()
    with torch.no_grad():
        baseline = card.run()[0].clamp_min(0).numpy()
        maximum = float(np.max(np.abs(baseline-ref['baseline'])))
        for name,eps in (('full',.02),('half',.01)):
            card.theta = eps*direction
            plus = card.run()[0].clamp_min(0).numpy()
            card.theta = -eps*direction
            minus = card.run()[0].clamp_min(0).numpy()
            response = (plus-minus)/(2*eps)
            saves[f'response_{name}'] = response
            row = dict(epsilon=eps,response_relative=rel(response[-1],ref[f'response_{name}'][-1]),
                       endpoint_max_abs=max(float(np.max(np.abs(plus-ref[f'plus_{name}']))),
                                            float(np.max(np.abs(minus-ref[f'minus_{name}'])))))
            rows.append(row)
            print(json.dumps(dict(rank=args.rank,**row)),flush=True)
    with torch.no_grad(),forward_ad.dual_level():
        scalar = forward_ad.make_dual(torch.zeros((),dtype=torch.float64),torch.ones((),dtype=torch.float64))
        card.theta = scalar*direction
        _, tangent = forward_ad.unpack_dual(card.run()[0].clamp_min(0))
        tangent = tangent.numpy()
    saves['tangent'] = tangent
    errors = [rel(tangent[-1],ref[f'response_{name}'][-1]) for name in ('full','half')]
    passed = maximum < 2e-6 and max(r['response_relative'] for r in rows) <= .10 and max(errors) <= .05 and npr['half_relative'] <= .05
    result = dict(rank=args.rank,zero_action_cpu_max_abs=maximum,rows=rows,
                  tangent_vs_numpy_relative=errors,numpy_half_relative=npr['half_relative'],
                  fidelity_pass=passed,wall_s=time.time()-t0,
                  max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  qualification='Target-free full CPU coherent-state derivative check. A failed cell/noise check must be diagnosed before fitting; not a score or globally smooth dynamics claim.')
    np.savez(arrays,**saves)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
