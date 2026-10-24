"""Both full Phase2 bank trajectories, no labels; fixed f32 fidelity gate."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from centered_bank_autodiff_v1 import CenteredBankFullCard

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device',choices=('cpu','mps'),default='cpu')
    ap.add_argument('--dtype',choices=('f32','f64'),default='f32')
    args = ap.parse_args()
    assert not (args.device=='mps' and args.dtype=='f64')
    out = HERE/f'centered_bank_baseline_{args.device}_{args.dtype}_v1.json'
    assert not out.exists() and not out.with_suffix('.npz').exists()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0') == '0'
    assert 'threadpoolctl' not in __import__('sys').modules
    small = json.loads((HERE/f'centered_torch_numerics_{args.device}_small_v1.json').read_text())
    assert small['passed']
    assert small['source_sha256'] == hashlib.sha256((HERE/'centered_torch_numerics_v1.py').read_bytes()).hexdigest()
    parent = json.loads((HERE/'centered_parent_numerics_v1.json').read_text())
    assert parent['passed']
    torch.set_num_threads(3)
    dtype = torch.float32 if args.dtype=='f32' else torch.float64
    rows,outputs = [],{}
    start = time.time()
    for rank in (35,100):
        ref = np.load(HERE/'centered_parent_numerics_v1.npz')[f'baseline_{rank}']
        # Old target-free local-cell receipt certifies exact baseline masks.
        mask_receipt = json.loads((HERE/f'full_carrier_local_cell_r{rank}_v1.json').read_text())
        assert mask_receipt['baseline_max_abs'] == 0
        masks = np.load(HERE/f'full_carrier_local_cell_r{rank}_v1.npz')['baseline_masks'][:30]
        card = CenteredBankFullCard(rank,args.device,dtype,audit_selections=True)
        means = []
        with torch.no_grad():
            state = card.initial()
            control = torch.zeros(card.n,device=card.device,dtype=dtype)
            for l in range(card.depth):
                state = card.step(l,state,control,card.P0)
                m = state[0].clamp_min(0).cpu().to(torch.float64).numpy().copy()
                means.append(m)
                print(json.dumps(dict(rank=rank,layer=l,max_mean_error=float(np.max(np.abs(m-ref[l]))),
                                      columns=state[2].shape[1],wall_s=time.time()-start)),flush=True)
        mean = np.stack(means)
        assert len(card.selections)==30
        memberships = [bool(np.array_equal(masks[j],row[1])) for j,row in enumerate(card.selections)]
        changed_counts = [int(np.count_nonzero(masks[j]!=row[1])) for j,row in enumerate(card.selections)]
        error = mean-ref
        maximum = float(np.max(np.abs(error)))
        row = dict(rank=rank,max_mean_error=maximum,layer_rms=np.sqrt(np.mean(error*error,axis=1)).tolist(),
                   membership_equal=memberships,membership_changed_counts=changed_counts,
                   passed=maximum<=2e-6 and all(memberships))
        rows.append(row)
        outputs[f'mean_{rank}'] = mean
        outputs[f'masks_{rank}'] = np.stack([x[1] for x in card.selections])
        for j,selection in enumerate(card.selections):
            outputs[f'indices_{rank}_{j}'] = selection[0]
        print(json.dumps(row),flush=True)
        del card,state,control,means
        gc.collect()
        if args.device=='mps':
            torch.mps.synchronize()
            torch.mps.empty_cache()
    np.savez(out.with_suffix('.npz'),**outputs)
    result = dict(passed=all(r['passed'] for r in rows),records=rows,device=args.device,dtype=args.dtype,
                  output_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
                  numerics_source_sha256=small['source_sha256'],
                  card_source_sha256=hashlib.sha256((HERE/'centered_bank_autodiff_v1.py').read_bytes()).hexdigest(),
                  target_access=False,fallback_enabled=False,wall_s=time.time()-start,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification='Full dynamic centered Torch bank with frozen P0 and scalar finalQ versus NumPy baseline on35/100. Membership checked; orders archived but not independently compared. No labels, fitting, full derivative, candidate or score claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
    assert result['passed'],result


if __name__=='__main__':
    main()
