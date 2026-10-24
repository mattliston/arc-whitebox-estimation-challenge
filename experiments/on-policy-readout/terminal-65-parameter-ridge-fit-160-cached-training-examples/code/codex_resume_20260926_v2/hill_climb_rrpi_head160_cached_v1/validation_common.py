"""Shared, unchanged pooled-score math and fixed cached-head validation checks."""
from pathlib import Path
import hashlib, json, math
import numpy as np
D=Path(__file__).resolve().parent
def pin(path):
 p=Path(path).resolve();raw=p.read_bytes()
 return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def checked(record):
 assert pin(record['path'])==record,record['path']
 return Path(record['path']).read_bytes()
def once(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def metrics(base, child):
    assert base and len(base) == len(child)
    for rows in (base, child):
        assert len({r['rank'] for r in rows}) == len(rows)
        for row in rows:
            assert all(math.isfinite(row[k]) for k in ('mse', 'adjusted', 'util', 'cpu_s', 'flops'))
            assert row['mse'] >= 0 and row['cpu_s'] > 0 and 0 < row['flops'] <= 2**41
            assert row['util'] == row['flops'] / 2**41
            assert row['adjusted'] == row['mse'] * max(.1, row['util'])
    assert all(r['mse'] > 0 for r in base)
    base = {r['rank']: r for r in base}; child = {r['rank']: r for r in child}
    ranks = sorted(base); assert ranks == sorted(child)
    keys = ('mse', 'adjusted', 'util', 'cpu_s')
    b = np.array([[base[r][k] for k in keys] for r in ranks])
    c = np.array([[child[r][k] for k in keys] for r in ranks])
    indices = np.random.default_rng(20260926).integers(0, len(ranks), size=(50000, len(ranks)))
    ci = np.quantile(c[indices, :2].mean(1)/b[indices, :2].mean(1)-1, [.025, .975], axis=0)*100
    per = [dict(rank=r, raw_pct=100*(child[r]['mse']/base[r]['mse']-1),
        adjusted_pct=100*(child[r]['adjusted']/base[r]['adjusted']-1),
        util_pct=100*(child[r]['util']/base[r]['util']-1)) for r in ranks]
    return dict(n=len(ranks), ranks=ranks, raw_pct=float(100*(c[:, 0].mean()/b[:, 0].mean()-1)),
        adjusted_pct=float(100*(c[:, 1].mean()/b[:, 1].mean()-1)), raw_ci95_pct=ci[:, 0].tolist(),
        adjusted_ci95_pct=ci[:, 1].tolist(), adjusted_wins=int((c[:, 1] < b[:, 1]).sum()),
        raw_wins=int((c[:, 0] < b[:, 0]).sum()), base_mean_adjusted=float(b[:, 1].mean()),
        candidate_mean_adjusted=float(c[:, 1].mean()), mean_util_pct=float(100*(c[:, 2].mean()/b[:, 2].mean()-1)),
        mean_cpu_pct=float(100*(c[:, 3].mean()/b[:, 3].mean()-1)),
        bill_pct=100*(sum(r['flops'] for r in child.values())/sum(r['flops'] for r in base.values())-1),
        max_per_network_adjusted_regression_pct=max(r['adjusted_pct'] for r in per), per_network=per)
def gates(result, limits, additional=None):
 out=dict(raw=result['raw_pct']<=limits['raw_pct_max'],
          adjusted=result['adjusted_pct']<=limits['adjusted_pct_max'],
          wins=result['adjusted_wins']>=limits['wins_min'],
          max_regression=result['max_per_network_adjusted_regression_pct']<=limits['max_regression_pct'])
 if 'ci_upper_max' in limits:out['ci']=result['adjusted_ci95_pct'][1]<limits['ci_upper_max']
 if limits.get('additional6_gain_required'):
  assert additional is not None
  out['additional6']=additional['adjusted_pct']<0
 return out
def configuration(path,digest):
 assert pin(path)['sha256']==digest
 c=json.loads(Path(path).read_text())
 assert c['schema']=='head160-native-validation-v1'
 for rec in c['pins']:checked(rec)
 checked(c['candidate']);checked(c['baseline'])
 protocol=json.loads(checked(c['protocol']))
 assert protocol['root_admitted'] is True and protocol['method'] == 'cached-S31-terminal65-ridge-transfer'
 fit_binding(c, protocol)
 assert c['research_parent'] == protocol['baseline'] and c['baseline'] == protocol['official']
 checked(c['research_parent'])
 assert c['ranks']==protocol['validation']['screen2']+protocol['validation']['additional6']
 assert c['ranks']==[37,39,42,48,59,63,66,67]
 assert not set(c['ranks'])&set(protocol['training_ranks'])
 assert c['gates2']==protocol['validation']['screen2_gates']
 assert c['gates8']==protocol['validation']['screen8_gates']
 assert c['guard_limits']==dict(seconds=300,rss_bytes=7500000000,added_swap_bytes=0)
 index=json.loads(checked(c['baseline_index']))
 assert index['ranks']==c['ranks'] and index['source']==c['baseline']
 assert index['actual_block_order'][:8]==c['ranks']
 assert len(index['data']) == len(c['data']) == len(c['ranks']) == 8
 assert [x['rank'] for x in index['data']] == [x['rank'] for x in c['data']] == c['ranks']
 data_review=json.loads(checked(index['data_review']))
 validation_data={x['rank']:x for x in data_review['validation_data']}
 for item in c['data']:
  reference=validation_data[item['rank']]
  assert item['weights'] == reference['weights'] and item['truth'] == reference['truth']
 for key in ['source_driver','guard_source','broad_config','block_job','block_guard','manifest','bills','source','official_bank']:checked(index[key])
 manifest=json.loads(checked(index['manifest']));job=json.loads(checked(index['block_job']))
 assert manifest['candidate'] == c['baseline']['path'] and manifest['sha256'] == c['baseline']['sha256']
 assert manifest['ranks'] == index['actual_block_order'] and manifest['runtime'] == index['runtime'] == c['runtime']
 assert manifest['numpy'] == '2.4.6' and manifest['thread_env'] == index['thread_env']
 assert all(v == '1' for v in index['thread_env'].values())
 assert job['passed'] is True and job['guard_exit_code'] == 0
 assert job['candidate'] == c['baseline'] and job['guard'] == index['block_guard']
 assert job['command'] == index['command'] and job['config'] == index['broad_config']
 bg=json.loads(checked(index['block_guard']))
 assert bg['completed'] and bg['whole_group_retired'] and not bg['remaining_rows'] and bg['exit_code']==0 and bg['error'] is None and bg['max_swap_growth_bytes']==0 and bg['max_rss_bytes']<7500000000
 assert bg['command']==index['command']
 assert 0 < bg['elapsed_s'] <= 420 and bg['rss_trigger_bytes'] == 7500000000
 rows=[json.loads(s) for s in checked(index['bills']).decode().splitlines() if s.strip()]
 assert [r['rank'] for r in rows] == index['actual_block_order']
 for item,row in zip(index['data'],rows):
  assert item['saved_row']==row
  for key in ('prediction','truth','driver_truth'):checked(item[key])
  assert item['truth']['sha256'] == item['driver_truth']['sha256']
  assert item['truth'] == validation_data[item['rank']]['truth']
  assert row['util'] == row['flops'] / 2**41
  assert row['adjusted'] == row['mse'] * max(.1, row['util'])
 return c,index,rows[:8]


def fit_binding(c, protocol):
 fit_protocol=json.loads(checked(c['fit_protocol']))
 assert c['fit_protocol'] == protocol['fit_protocol']
 result=json.loads(checked(c['fit_result']))
 assert result['completed'] is True and result['passed'] is True and result['validation_read'] is False
 assert result['protocol'] == c['fit_protocol']
 assert result['data_index'] == c['fit_data_index']
 assert result['ridge'] == 1e-3 and result['ranks'] == protocol['training_ranks']
 exported=json.loads(checked(c['export_manifest']))
 assert exported['passed'] is True and exported['candidate'] == c['candidate']
 assert exported['baseline'] == c['research_parent']
 assert exported['protocol'] == c['fit_protocol'] and exported['fit_result'] == c['fit_result']
 assert exported['exporter'] == c['fit_driver']
 assert result['candidate'] == exported['candidate'] and result['baseline'] == c['research_parent']
 assert result['allowed_parameter_count'] == exported['allowed_parameter_count'] == 65
 assert result['source_byte_inverse_exact'] is True and exported['source_byte_inverse_exact'] is True
 assert result['all_other_decoded_parameter_bytes_exact'] is True and exported['all_other_decoded_parameter_bytes_exact'] is True
 assert result['executable_ast_unchanged'] is True and exported['executable_ast_unchanged'] is True
 assert result['surrogate_proposal'] is True and result['native_readout_replay_certified'] is False
 assert result['fit_diagnostics_can_admit'] is False and result['actual_native_score_measured'] is False
 checked(c['fit_data_index']);checked(c['fit_driver'])
 assert len(set(result['ranks'])) == len(result['ranks']) == 160
 assert not set(result['ranks']) & set(c['ranks'])
 fitguard=json.loads(checked(c['fit_guard']))
 assert Path(c['fit_guard']['path']) == D/'fit_guard.json'
 plan=json.loads(checked(c['validation_plan']))
 assert c['fit_command']==plan['fit_command'] and c['fit_limits']==plan['fit_limits']
 assert c['fit_command'] == [c['runtime'],'-B',c['fit_driver']['path'],'--protocol-sha',c['fit_protocol']['sha256']]
 assert Path(c['runtime']).resolve() == Path(fit_protocol['runtime']['path']).resolve()
 assert fitguard['command']==c['fit_command']
 assert fitguard['completed'] is True and fitguard['whole_group_retired'] is True
 assert fitguard['exit_code']==0 and fitguard['error'] is None and fitguard['remaining_rows']==[]
 assert fitguard['rss_trigger_bytes']==c['fit_limits']['rss_bytes']
 assert 0 < fitguard['max_rss_bytes'] < c['fit_limits']['rss_bytes']
 assert 0 < fitguard['elapsed_s'] <= c['fit_limits']['seconds']
 assert fitguard['max_swap_growth_bytes']==c['fit_limits']['added_swap_bytes']==0
 return result,exported
