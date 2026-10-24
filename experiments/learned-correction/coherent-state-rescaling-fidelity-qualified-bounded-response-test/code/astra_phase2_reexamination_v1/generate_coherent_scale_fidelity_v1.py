"""NumPy zero-action parity and dynamic secants; no scoring targets."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from coherent_scale_reference_runner_v1 import Reference, selection_changes

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, choices=(35,100), required=True)
    args = ap.parse_args()
    out = HERE/f'coherent_scale_numpy_fidelity_r{args.rank}_v1.json'
    arrays = out.with_suffix('.npz')
    assert not out.exists() and not arrays.exists()
    assert json.loads((HERE/'coherent_central_scale_algebra_v1.json').read_text())['passed']
    ref = Reference(args.rank)
    original = np.load(HERE/f'fullpair_scalar_final_closedloop_r{args.rank}_v1.npz')['mean']
    direction = np.random.default_rng(966350).normal(size=(3,4))
    direction /= np.linalg.norm(direction)
    rows, saves = [], dict(direction=direction)
    t0 = time.time()
    with threadpool_limits(limits=3):
        base, bs, scales = ref.run(np.zeros((3,4)))
        maximum = float(np.max(np.abs(base-original)))
        assert maximum == 0, maximum
        assert np.array_equal(scales, np.ones_like(scales))
        saves['baseline'] = base
        for name, eps in (('full', .02), ('half', .01)):
            plus, ps, pscale = ref.run(eps*direction)
            minus, ms, mscale = ref.run(-eps*direction)
            response = (plus-minus)/(2*eps)
            prefix = max(float(np.max(np.abs(plus[:2]-base[:2]))),
                         float(np.max(np.abs(minus[:2]-base[:2]))))
            assert prefix == 0
            saves[f'plus_{name}'], saves[f'minus_{name}'] = plus, minus
            saves[f'response_{name}'] = response
            rows.append(dict(epsilon=eps, changes=selection_changes(bs,ps,ms),
                             prefix_through_l1_max_abs=prefix,
                             response_rms=float(np.sqrt(np.mean(response[-1]**2))),
                             scale_minimum=min(float(pscale.min()),float(mscale.min())),
                             scale_maximum=max(float(pscale.max()),float(mscale.max()))))
            print(json.dumps(dict(rank=args.rank,**{k:v for k,v in rows[-1].items() if k != 'changes'})),flush=True)
    a, b = saves['response_full'][-1], saves['response_half'][-1]
    half = float(np.linalg.norm(a-b)/np.linalg.norm(b))
    result = dict(rank=args.rank, zero_action_max_abs=maximum, rows=rows,
                  half_relative=half, wall_s=time.time()-t0,
                  qualification='Target-free coherent-state NumPy parity and dynamic-response certificate only; selection changes are recorded, not removed. No scoring label or improvement claim.')
    np.savez(arrays,**saves)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rank=args.rank,zero_action_max_abs=maximum,half_relative=half,wall_s=result['wall_s'])),flush=True)


if __name__ == '__main__':
    main()
