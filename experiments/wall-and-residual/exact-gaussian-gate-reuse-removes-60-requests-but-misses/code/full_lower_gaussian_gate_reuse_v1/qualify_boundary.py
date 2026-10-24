import sys
import numpy as np
from common import HERE, read, write, pin, verify, sha, seal_check

seal_check('boundary_sources_sealed.json')
native = read(HERE/'native_qualified.json')
assert native['passed']
removed = native['expected_removed_flops']
changed_ops = {'maximum','sqrt','true_divide','stats.norm.cdf'}

def equal(a, b):
    x = np.load(a); y = np.load(b)
    assert x.shape == y.shape == (16,1024) and x.dtype == y.dtype == np.float32
    assert x.tobytes() == y.tobytes() and np.isfinite(x).all() and (x>=0).all()

def receipt(p):
    r = read(p)
    assert r['completed'] and not r['truth_accessed'] and not r['numpy_available']
    assert not r['profiling'] and not r['accounting_modified']
    return r

if sys.argv[1] == 'pairs':
    source = read(HERE/'pairs_completed.json')
    assert source['completed'] and len(source['rows']) == 10
    for row in source['rows']:
        verify(row['receipt'])
    rows = []
    for index, rank in enumerate([35,100,37,39,42]):
        a = receipt(HERE/f'pair{index:02d}_{rank}_candidate/completed.json')
        b = receipt(HERE/f'pair{index:02d}_{rank}_parent/completed.json')
        assert a['candidate_sha256'] == native['candidate']['sha256']
        assert b['candidate_sha256'] == native['parent']['sha256']
        assert len(a['records']) == len(b['records']) == 1
        a = a['records'][0]; b = b['records'][0]
        assert a['rank'] == b['rank'] == rank and a['includes_initialization'] and b['includes_initialization']
        assert a['weights_sha256'] == b['weights_sha256']
        equal(a['output'],b['output'])
        own, parent = a['summary'], b['summary']
        assert parent['flops_used']-own['flops_used'] == removed
        assert own['operations'].keys() == parent['operations'].keys()
        deltas = {}
        for op in parent['operations']:
            ac = own['operations'][op]; bc = parent['operations'][op]
            difference = bc['calls']-ac['calls']
            assert difference == (15 if op in changed_ops else 0), (rank,op,difference)
            cost_difference = bc['flop_cost']-ac['flop_cost']
            assert cost_difference > 0 if op in changed_ops else cost_difference == 0
            if difference:
                deltas[op] = dict(removed_calls=difference,removed_flops=cost_difference)
        if rank in [35,100]:
            n = read(HERE/('enabled_triple' if rank==35 else 'enabled_cold100')/'result.json')['records'][0]
            equal(a['output'],n['output']); assert own['flops_used'] == n['flops']
        ar = own['residual_wall_time_s']; br = parent['residual_wall_time_s']
        assert ar > 0 and br > 0
        rows.append(dict(rank=rank,all16_bytes_exact=True,exact_operation_changes=deltas,
            candidate_flops=own['flops_used'],parent_flops=parent['flops_used'],
            candidate_residual_s=ar,parent_residual_s=br,residual_ratio=ar/br,
            candidate_wall_s=a['wall_s'],parent_wall_s=b['wall_s'],wall_ratio=a['wall_s']/b['wall_s'],
            candidate_output=pin(a['output']),parent_output=pin(b['output'])))
    ratio = sum(x['candidate_residual_s'] for x in rows)/sum(x['parent_residual_s'] for x in rows)
    wins = sum(x['residual_ratio'] < 1 for x in rows)
    maximum = max(x['candidate_residual_s'] for x in rows)
    worst = max(x['residual_ratio'] for x in rows)
    wall = sum(x['candidate_wall_s'] for x in rows)/sum(x['parent_wall_s'] for x in rows)
    wall_worst = max(x['wall_ratio'] for x in rows)
    passed = ratio <= .98 and wins >= 4 and maximum <= .25 and worst <= 1.10 and wall <= 1.05 and wall_worst <= 1.10
    result = dict(passed=passed,residual_ratio=ratio,residual_wins=wins,maximum_candidate_residual_s=maximum,
        worst_residual_ratio=worst,wall_ratio=wall,worst_wall_ratio=wall_worst,rows=rows,
        all16_bytes_exact=True,removed_requests_per_call=60,removed_flops_per_call=removed,
        prediction_only_scope=True,platform_runtime_gain_proved=False,targets_accessed=False)
    write(HERE/'pairs_qualified.json',result)
    write(HERE/'outcome.json',dict(stage='five_pairs',passed=passed,study_closed=not passed,
        warm_client_pending=passed,uploaded=False))
    print({k:v for k,v in result.items() if k!='rows'},flush=True)
elif sys.argv[1] == 'warm':
    assert read(HERE/'pairs_qualified.json')['passed']
    r = receipt(HERE/'client_triple/completed.json')
    assert r['candidate_sha256'] == native['candidate']['sha256']
    own = r['records']; base = read(HERE/'enabled_triple/result.json')['records']
    assert len(own) == len(base) == 3
    for a,b in zip(own,base):
        assert a['rank'] == b['rank'] and a['includes_initialization'] == b['includes_initialization']
        equal(a['output'],b['output']); assert a['summary']['flops_used'] == b['flops']
    assert sha(own[0]['output']) == sha(own[2]['output'])
    cold100 = receipt(HERE/'pair01_100_candidate/completed.json')['records'][0]
    equal(own[1]['output'],cold100['output'])
    residuals = [r['summary']['residual_wall_time_s'] for r in own]
    passed = all(0 < r <= .25 for r in residuals)
    write(HERE/'boundary_qualified.json',dict(passed=passed,pairs=pin(HERE/'pairs_qualified.json'),
        all16_native_client_bytes_and_bills_exact=True,repeat35_exact=True,warm_cold100_exact=True,
        warm_residuals_s=residuals,removed_flops_per_call=removed,candidate=native['candidate'],
        parent=native['parent'],targets_accessed=False,platform_runtime_gain_proved=False))
    write(HERE/'warm_outcome.json',dict(stage='warm_client',passed=passed,study_closed=not passed,
        release_pending=passed,uploaded=False))
    print(dict(passed=passed,warm_residuals_s=residuals),flush=True)
else:
    raise ValueError(sys.argv[1])
