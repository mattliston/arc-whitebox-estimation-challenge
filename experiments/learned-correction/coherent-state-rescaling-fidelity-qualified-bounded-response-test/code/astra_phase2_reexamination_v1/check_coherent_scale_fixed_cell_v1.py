"""Independent local selection-branch certificate for coherent rescaling."""
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
    ap.add_argument('--rank',type=int,choices=(35,100),required=True)
    args = ap.parse_args()
    out = HERE/f'coherent_scale_fixed_cell_r{args.rank}_v1.json'
    arrays = out.with_suffix('.npz')
    assert not out.exists() and not arrays.exists()
    npr = np.load(HERE/f'coherent_scale_numpy_fidelity_r{args.rank}_v1.npz')
    tangent = np.load(HERE/f'coherent_scale_autodiff_r{args.rank}_v1.npz')['tangent']
    ref = Reference(args.rank)
    rows,saves = [],{}
    t0 = time.time()
    with threadpool_limits(limits=3):
        base,bs,_ = ref.run(np.zeros((3,4)))
        maximum = float(np.max(np.abs(base-npr['baseline'])))
        assert maximum == 0
        for name,eps in (('full',.02),('half',.01)):
            plus,ps,_ = ref.run(eps*npr['direction'],fixed=bs)
            minus,ms,_ = ref.run(-eps*npr['direction'],fixed=bs)
            assert not selection_changes(bs,ps,ms)
            response = (plus-minus)/(2*eps)
            saves[f'response_{name}'] = response
            error = float(np.linalg.norm(response[-1]-tangent[-1])/np.linalg.norm(tangent[-1]))
            rows.append(dict(epsilon=eps,tangent_relative=error))
            print(json.dumps(dict(rank=args.rank,**rows[-1])),flush=True)
    a,b = saves['response_full'][-1],saves['response_half'][-1]
    half = float(np.linalg.norm(a-b)/np.linalg.norm(b))
    result = dict(rank=args.rank,baseline_max_abs=maximum,rows=rows,
                  half_relative=half,branch_fidelity_pass=half <= .05 and max(r['tangent_relative'] for r in rows) <= .05,
                  wall_s=time.time()-t0,
                  qualification='Target-free fixed baseline membership AND ordering certificate only. Actual dynamics retain changing selections; original failed global flag is not overwritten. No target, fit or score claim.')
    np.savez(arrays,**saves)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
