"""Reuse authenticated parent errors only after all14 predictions match exactly."""
import hashlib
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'read5_packet_v1'
read = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
build = read(HERE / 'build.json')
panel = read(HERE / 'panel/completed.json')
assert panel['completed'] and panel['candidate_sha256'] == build['candidate_sha256']
assert sha(HERE / 'candidate.py') == build['candidate_sha256']
old_score = read(PARENT / 'panel_score.json')
old = {r['rank']: r for r in old_score['rows']}
rows = []
for record in panel['records']:
    assert record['all16_byte_exact']
    call = record['call']
    assert sha(call['output']) == call['output_sha256']
    prior = old[record['rank']]
    parent_call = read(record['parent_result'])['records'][0]
    assert sha(parent_call['output']) == call['output_sha256']
    assert call['weights_sha256'] == parent_call['weights_sha256']
    assert prior['candidate']['flops'] == parent_call['flops']
    mult = max(.1, call['flops'] / 2**41)
    candidate = dict(raw=prior['candidate']['raw'], raw_nc=prior['candidate']['raw_nc'],
                     flops=call['flops'], multiplier=mult,
                     adjusted=prior['candidate']['raw'] * mult,
                     adjusted_nc=prior['candidate']['raw_nc'] * mult)
    rows.append(dict(rank=record['rank'], all16_byte_exact=True,
                     truth_sha256=prior['truth_sha256'], noise_mse=prior['noise_mse'],
                     candidate=candidate, read5=prior['candidate'], cost150=prior['cost150'],
                     adjusted_nc_ratio=candidate['adjusted_nc'] / prior['candidate']['adjusted_nc']))
assert len(rows) == 14 and all(r['adjusted_nc_ratio'] < 1 for r in rows)
groups = {}
for name, selected in [('public8', rows[:8]), ('fresh6', rows[8:]), ('all14', rows)]:
    block = {who: {metric: mean(r[who][metric] for r in selected)
                   for metric in ('raw', 'raw_nc', 'flops', 'adjusted', 'adjusted_nc')}
             for who in ('candidate', 'read5', 'cost150')}
    block['adjusted_nc_ratio'] = block['candidate']['adjusted_nc'] / block['read5']['adjusted_nc']
    block['wins'] = sum(r['adjusted_nc_ratio'] < 1 for r in selected)
    groups[name] = block
ratio = groups['all14']['candidate']['adjusted_nc'] / groups['all14']['cost150']['adjusted_nc']
projection = 4.015535762022727e-9 * ratio
receipt = dict(candidate_sha256=build['candidate_sha256'], rows=rows, groups=groups,
               reused_parent_errors_after_byte_parity=True, raw_targets_opened=False,
               parent_score_sha256=sha(PARENT / 'panel_score.json'),
               official_bank_ratio_projection=projection,
               projection_caveat='Existing local14 panel, with noise correction; not untouched validation or an official result.',
               panel_gate_passed=all(g['adjusted_nc_ratio'] < 1 for g in groups.values()))
with (HERE / 'panel_score.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps({k: v for k, v in receipt.items() if k != 'rows'}), flush=True)
