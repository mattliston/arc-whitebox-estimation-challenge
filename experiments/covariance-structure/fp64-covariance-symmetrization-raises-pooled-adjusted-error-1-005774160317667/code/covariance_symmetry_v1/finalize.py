"""Record the exact fixed development outcome, without changing its gate."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
D = HERE.parent
A = D.parent/'astra_phase2_reexamination_v1'
ROOT = Path('/Users/matt/arc-white-box-estimation')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')

seal=read(HERE/'source_sealed.json')
for name,expected in seal['files_sha256'].items():
    assert sha(HERE/name)==expected,name
assert sha(D/seal['native_runner'])==seal['native_runner_sha256']
for name,expected in read(HERE/'source_review.json')['sources_sha256'].items():
    assert sha(A/name)==expected,name
build=read(HERE/'build.json')
assert build['whole_source_inverse_byte_exact'] and build['exact_insertions']==1
assert sha(D/'lazy_rotation_objective_v1/candidate.py')==build['parent_sha256']
unit=read(HERE/'unit_v1/result.json')
assert unit['passed'] and unit['repeated_arrays_byte_exact'] and unit['repeated_bills_exact']
watches=[read(HERE/name) for name in ['unit_watch.json','dev_watch.json','cold100_watch.json','score_watch.json']]
for w in watches:
    assert w['completed'] and w['exit_code']==0 and w['error'] is None
    assert w['whole_group_retired'] and w['max_swap_growth_bytes']==0
    assert w['max_rss_bytes']<8_000_000_000
calls=[]
for directory in ['dev','cold100']:
    result=read(HERE/directory/'result.json')
    assert result['candidate_sha256']==build['candidate_sha256']
    assert result['runner_sha256']==seal['native_runner_sha256']
    assert not result['truth_accessed']
    for record in result['records']:
        assert sha(record['output'])==record['output_sha256']
        assert record['weight5_cache']['completed'] and record['weight5_cache']['numeric_cache_retired']
        calls.append(record)
assert len(calls)==4
assert calls[0]['output_sha256']==calls[2]['output_sha256']
assert calls[1]['output_sha256']==calls[3]['output_sha256']
score=read(HERE/'dev_score.json')
assert score['repeat_byte_exact'] and score['warm_cold100_byte_exact'] and score['first_layer_byte_exact']
assert [r['rank'] for r in score['rows']]==[35,100]
peak=max([w['max_rss_bytes'] for w in watches]+[r['max_rss_bytes'] for r in calls]+[unit['max_rss_bytes']])
assert peak<8_000_000_000
resources=dict(full_prediction_calls=4,guarded_stages=4,
               billed_prediction_flops=sum(r['flops'] for r in calls),
               peak_rss_bytes=peak,additional_swap_bytes=0,all_owned_groups_retired=True,
               total_guarded_wall_s=sum(w['elapsed_s'] for w in watches))
write_json(HERE/'resources.json',resources)
bank=read(D/'official_best_331599.json')
assert sha(D/'lazy_rotation_objective_v1/submission-lazy-objective-20260919.tar.gz')==bank['archive_sha256']
passed=score['development_gate_passed']
status='passed development; wider qualification required' if passed else 'closed negative at fixed development gate'
lines=['# Fixed covariance symmetry: '+status,'',
       'One insertion symmetrizes full Cpre/Bpre in FP64 before their consumers.',
       'The directed K21 plane, final diagonal path, rank384/age4, all learned',
       'parameters, Lazy schedule and kernels are unchanged. No PSD repair.',
       'This tests numerical rounding, not a new physical source.', '',
       '| Rank | Lazy raw MSE | Symmetry raw MSE | Raw ratio | Own-cold adjusted ratio |',
       '| --- | ---: | ---: | ---: | ---: |']
for row in score['rows']:
    lines.append(f"| {row['rank']} | {row['parent']['raw']:.16g} | {row['candidate']['raw']:.16g} | {row['raw_ratio']:.12g} | {row['adjusted_ratio']:.12g} |")
lines+=['',f"Pooled adjusted ratio: **{score['pooled_adjusted_ratio']:.16g}**.",
        'The fixed gate requires both raw/adjusted wins and pooledadjusted<=.99.',
        'These are two reused ordinary1B development networks, not independent',
        'generalization evidence or an official result. Own complete cold bills',
        'include all symmetry arithmetic and any changed selector geometry.','']
for row in score['rows']:
    increment=row['candidate']['flops']-row['parent']['flops']
    lines.append(f"Rank{row['rank']} cold FLOPs: {row['parent']['flops']} -> {row['candidate']['flops']} ({increment:+d}).")
lines+=['',
        'All seven explicit entrywise certificates, symmetry/idempotence,',
        'diagonal and quadratic-form checks pass; maximum scaled quadratic error',
        f"{max(x['quadratic_relative_scaled_error'] for x in unit['cases']):.16g}.",
        'Three malformed-input rejections, exact unit arrays/bills, complete',
        'network repeat/warm-cold100 bytes, first-layer parent mean equality,',
        'finite nonnegative outputs and cache retirement pass. The six sealed',
        'source/protocol files remain unchanged. Predictions and bills were',
        'sealed before opening the two pinned truth files.','',
        f"Four predictions total {resources['billed_prediction_flops']} FLOPs; peak RSS {peak} bytes.",
        'All four guarded stages exit0 and retire; zero added swap.','']
if passed:
    lines+=['This only authorizes further fixed local qualification, not a submission',
            'claim. Wider panel and actual client/contract/package checks are pending.','']
else:
    lines+=['No wider panel, fitted rescue, strength/location/precision/kernel sweep,',
            'production client, package or upload follows this failed variant.','']
lines += [f"Official bank remains Lazy #331599 adjusted {bank['adjusted']:.16g}.",
          'The4e-9 milestone is confirmed;3e-9,2e-9 and approximately1e-9 remain unmet.',
          'Future qualified Phase2 board uploads have the user’s standing preauthorization.','']
name='RESULT.md' if passed else 'NEGATIVE_RESULT.md'
with (HERE/name).open('x') as stream:stream.write('\n'.join(lines))
artifacts={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts}
size=sum((HERE/name).stat().st_size for name in artifacts)
assert size<10_000_000
closeout=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),status=status,
              goal_status='active',goal_achieved=False,official_bank=bank,
              candidate_sha256=build['candidate_sha256'],parent_sha256=build['parent_sha256'],
              source_inverse_byte_exact=True,unit_passed=True,development_gate_passed=passed,
              pooled_adjusted_ratio=score['pooled_adjusted_ratio'],
              raw_ratios=[r['raw_ratio'] for r in score['rows']],
              adjusted_ratios=[r['adjusted_ratio'] for r in score['rows']],
              resources=resources,artifact_bytes_before_closeout=size,artifacts_sha256=artifacts,
              remaining_disk_GiB=shutil.disk_usage(ROOT).free/2**30)
write_json(HERE/'closeout.json',closeout)
print(json.dumps({k:v for k,v in closeout.items() if k not in ['artifacts_sha256','official_bank']}))
