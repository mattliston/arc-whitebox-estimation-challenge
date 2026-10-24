"""Paired ordinary/NC scores; opened only after the candidate panel completes."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TRUTH = HERE.parent/'public_g19_attribution_v1'


def main():
    panel = json.loads((HERE/'cost75_panel_v1/completed.json').read_text())
    assert panel['completed'] and len(panel['records']) == 14
    parent_panel=json.loads((HERE/'cost150_panel_v1/completed.json').read_text())
    parent_by_rank={r['rank']:r['call'] for r in parent_panel['records']}
    assert parent_panel['candidate_sha256']=='f06314111d7f41a7c1fbcb8d888a509d0e46222cd31b5ab227ec1e334e582a1f'
    assert panel['candidate_sha256']=='5f66f94afdd2909584527a299c3989377d374289b8f5e1f379047b4176c7521c'
    rows = []
    for entry in panel['records']:
        rank = entry['rank']
        path = TRUTH/(f'fresh_{rank}_truth4M.npz' if rank>=3016500 else f'mps_truth_rank{rank:04d}_indep4M.npz')
        with np.load(path) as f:
            truth = f['mc_post'][15].astype(np.float64)
            noise = float(np.mean(f['mc_se'][15].astype(np.float64)**2))
        row = dict(rank=rank, truth_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), noise_mse=noise)
        arms = {'baseline': parent_by_rank[rank],
                'candidate': entry['call']}
        outputs = []
        for arm, call in arms.items():
            assert hashlib.sha256(Path(call['output']).read_bytes()).hexdigest() == call['output_sha256']
            pred = np.load(call['output']).astype(np.float64)
            assert pred.shape==(16,1024) and np.isfinite(pred).all() and (pred>=0).all()
            outputs.append(pred)
            mse = float(np.mean((pred[15]-truth)**2))
            row[arm] = dict(raw=mse, raw_nc=mse-noise, flops=call['flops'], multiplier=call['multiplier'],
                            adjusted=mse*call['multiplier'], adjusted_nc=(mse-noise)*call['multiplier'])
        row['prefix_delta_mse'] = float(np.mean((outputs[0][:15]-outputs[1][:15])**2))
        row['adjusted_nc_ratio'] = row['candidate']['adjusted_nc']/row['baseline']['adjusted_nc']
        rows.append(row)
    groups = {}
    for name, group in [('public8',rows[:8]),('virgin6',rows[8:]),('all14',rows)]:
        aggregate = {arm:{key:float(np.mean([r[arm][key] for r in group]))
                          for key in ('raw','raw_nc','flops','multiplier','adjusted','adjusted_nc')}
                     for arm in ('baseline','candidate')}
        aggregate['adjusted_nc_ratio'] = aggregate['candidate']['adjusted_nc']/aggregate['baseline']['adjusted_nc']
        aggregate['wins'] = sum(r['adjusted_nc_ratio']<1 for r in group)
        groups[name] = aggregate
    estimate = 4.015535762022727e-9*groups['all14']['adjusted_nc_ratio']
    result = dict(rows=rows, groups=groups, official_bank_ratio_projection=estimate,
                  projection_caveat='Paired local NC estimate only; not an official score or guaranteed threshold crossing.')
    result['both_panels_improve']=all(groups[g]['adjusted_nc_ratio']<1 for g in ('public8','virgin6'))
    result['largest_individual_nc_ratio']=max(r['adjusted_nc_ratio'] for r in rows)
    result['local_milestone_projection']=estimate<4e-9
    with (HERE/'cost75_panel_score.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(dict(groups=groups,official_bank_ratio_projection=estimate)),flush=True)


if __name__ == '__main__':
    main()
