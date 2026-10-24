"""Compare CPU conditional-birth secants/tangent against actual NumPy."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.autograd import forward_ad
from conditional_birth_autodiff_v1 import ConditionalBirthCard

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank',type=int,choices=(35,100),required=True)
    args = ap.parse_args()
    out = HERE/f'conditional_birth_autodiff_r{args.rank}_v1.json'
    arrays = out.with_suffix('.npz')
    assert not out.exists() and not arrays.exists()
    torch.set_num_threads(3)
    ref = np.load(HERE/f'conditional_birth_numpy_fidelity_r{args.rank}_v1.npz')
    npr = json.loads((HERE/f'conditional_birth_numpy_fidelity_r{args.rank}_v1.json').read_text())
    direction = torch.tensor(ref['direction'])
    card = ConditionalBirthCard(args.rank)
    t0 = time.time()
    with torch.no_grad():
        base = card.run()[0].clamp_min(0).numpy()
    maximum = float(np.max(np.abs(base-ref['baseline'])))
    rows,saves = [],{}
    def relative(a,b):
        return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
    with torch.no_grad():
        for name,eps in (('full',.05),('half',.025)):
            card.theta = eps*direction
            plus = card.run()[0].clamp_min(0)
            card.theta = -eps*direction
            minus = card.run()[0].clamp_min(0)
            response = ((plus-minus)/(2*eps)).numpy()
            target = ref[f'response_{name}']
            row = dict(epsilon=eps,response_relative=relative(response[-1],target[-1]),
                       maximum_endpoint_discrepancy=max(float(np.max(np.abs(plus.numpy()-ref[f'plus_{name}']))),float(np.max(np.abs(minus.numpy()-ref[f'minus_{name}'])))))
            rows.append(row)
            saves[f'response_{name}'] = response
            print(json.dumps(dict(rank=args.rank,**row)),flush=True)
    with torch.no_grad(),forward_ad.dual_level():
        scalar = forward_ad.make_dual(torch.zeros((),dtype=torch.float32),torch.ones((),dtype=torch.float32))
        card.theta = scalar*direction
        output = card.run()[0].clamp_min(0)
        _,tangent = forward_ad.unpack_dual(output)
        tangent = tangent.numpy()
    saves['tangent'] = tangent
    errors = [relative(tangent[-1],ref[f'response_{name}'][-1]) for name in ('full','half')]
    half = relative(saves['response_full'][-1],saves['response_half'][-1])
    passed = maximum < 2e-6 and max(r['response_relative'] for r in rows) <= .10 and max(errors) <= .10 and npr['finite_difference_half_relative'] <= .05
    receipt = dict(rank=args.rank,zero_head_cpu_max_abs=maximum,rows=rows,
                   forward_tangent_vs_numpy_relative=errors,
                   numpy_half_relative=npr['finite_difference_half_relative'],torch_half_relative=half,
                   fidelity_pass=passed,wall_s=time.time()-t0,
                   max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                   qualification='Target-free full-bank conditional-birth CPU secant/tangent check; no fit, score, candidate or billed inference claim. Failed cell/precision checks require diagnosis before fitting.')
    np.savez(arrays,**saves)
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main()
