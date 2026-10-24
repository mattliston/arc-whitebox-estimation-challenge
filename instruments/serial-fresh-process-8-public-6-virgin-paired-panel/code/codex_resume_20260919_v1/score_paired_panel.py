"""Independent-4M and noise-corrected panel scores after sealed acquisition."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'paired14_v1'
TRUTH = HERE.parent / 'public_g19_attribution_v1'


def main():
    completion = json.loads((OUT/'completed.json').read_text())
    assert completion['completed'] and completion['calls'] == 28
    result = dict(frame='independent 4M truths; nc subtracts mean(mc_se**2)', rows=[], groups={})
    for rank in completion['networks']:
        path = TRUTH/(f'fresh_{rank}_truth4M.npz' if rank >= 3016500 else f'mps_truth_rank{rank:04d}_indep4M.npz')
        with np.load(path) as labels:
            truth = labels['mc_post'][15].astype(np.float64)
            noise = float(np.mean(labels['mc_se'][15].astype(np.float64)**2))
        row = dict(rank=rank, truth_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), noise_mse=noise)
        predictions = []
        for arm in ('baseline', 'scalar'):
            receipt = json.loads((OUT/f'{arm}_{rank}'/'result.json').read_text())
            call = receipt['records'][0]
            raw = Path(call['output']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == call['output_sha256']
            pred = np.load(call['output']).astype(np.float64)
            predictions.append(pred)
            mse = float(np.mean((pred[15]-truth)**2))
            row[arm] = dict(raw=mse, raw_nc=mse-noise, flops=call['flops'],
                            multiplier=call['multiplier'], adjusted=mse*call['multiplier'],
                            adjusted_nc=(mse-noise)*call['multiplier'])
        assert np.array_equal(predictions[0][:15], predictions[1][:15])
        row['prefix_byte_exact'] = True
        row['adjusted_nc_ratio'] = row['scalar']['adjusted_nc']/row['baseline']['adjusted_nc']
        result['rows'].append(row)
    for group, rows in [('public8',result['rows'][:8]),('virgin6',result['rows'][8:]),('all14',result['rows'])]:
        aggregate = {arm:{k:float(np.mean([r[arm][k] for r in rows])) for k in
                         ('raw','raw_nc','flops','multiplier','adjusted','adjusted_nc')}
                     for arm in ('baseline','scalar')}
        aggregate['adjusted_nc_ratio'] = aggregate['scalar']['adjusted_nc']/aggregate['baseline']['adjusted_nc']
        aggregate['wins'] = sum(r['adjusted_nc_ratio'] < 1 for r in rows)
        result['groups'][group] = aggregate
    with (HERE/'paired14_score.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result['groups']), flush=True)


if __name__ == '__main__':
    main()
