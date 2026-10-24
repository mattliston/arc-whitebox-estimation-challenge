"""Authenticate the fixed development result and preserve its exact scope."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
D=HERE.parent
ROOT=Path('/Users/matt/arc-white-box-estimation')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2)
        f.write('\n')


seal=read(HERE/'sources_sealed.json')
for name,digest in seal['files'].items():assert sha(HERE/name)==digest,name
for name,digest in seal['external'].items():assert sha(name)==digest,name
for row in read(HERE/'source_review.json')['sources']:
    assert sha(row['path'])==row['sha256'],row['path']
build=read(HERE/'build.json')
assert build['whole_source_inverse_byte_exact'] and build['final_estimator_call_edits']==1
assert sha(D/'lazy_rotation_objective_v1/candidate.py')==build['parent_sha256']
unit=read(HERE/'unit_v1/result.json')
assert unit['passed'] and unit['repeated_arrays_byte_exact'] and unit['repeated_bills_exact']
watches=[read(HERE/name) for name in ['unit_watch.json','dev_watch.json','cold100_watch.json','score_watch.json']]
for w in watches:
    assert w['completed'] and w['exit_code']==0 and w['error'] is None
    assert w['whole_group_retired'] and w['max_swap_growth_bytes']==0
    assert w['max_rss_bytes']<8_000_000_000
calls=[]
for dirname in ['dev','cold100']:
    result=read(HERE/dirname/'result.json')
    assert result['candidate_sha256']==build['candidate_sha256']
    assert result['runner_sha256']==seal['external'][str(D/'weight5_reuse_v1/native.py')]
    assert not result['truth_accessed']
    for r in result['records']:
        assert sha(r['output'])==r['output_sha256']
        assert r['weight5_cache']['completed'] and r['weight5_cache']['numeric_cache_retired']
        assert r['multiplier']==max(.1,r['flops']/2**41)
        calls.append(r)
assert len(calls)==4
assert calls[0]['output_sha256']==calls[2]['output_sha256']
assert calls[1]['output_sha256']==calls[3]['output_sha256']
score=read(HERE/'dev_score.json')
assert score['repeat_byte_exact'] and score['warm_cold100_byte_exact']
assert score['first_layer_absolute_gate_passed']
assert [r['rank'] for r in score['rows']]==[35,100]
peak=max([w['max_rss_bytes'] for w in watches]+[r['max_rss_bytes'] for r in calls]+[unit['max_rss_bytes']])
assert peak<8_000_000_000
resources=dict(full_prediction_calls=4,guarded_stages=4,
               billed_prediction_flops=sum(r['flops'] for r in calls),
               peak_rss_bytes=peak,additional_swap_bytes=0,all_owned_groups_retired=True,
               total_guarded_wall_s=sum(w['elapsed_s'] for w in watches))
write_json(HERE/'resources.json',resources)
bank=read(D/'official_best_331599.json')
assert bank['submission_id']==331599
assert sha(D/'lazy_rotation_objective_v1/submission-lazy-objective-20260919.tar.gz')==bank['archive_sha256']
passed=score['development_gate_passed']
status='passed development; wider qualification required' if passed else 'closed negative at fixed development gate'
lines=['# Incoming-column normalization: '+status,'',
       'One fixed positive-diagonal gauge normalizes hidden incoming columns to',
       'squared norm2, with final scale1. FP64 scale arithmetic and FP32 stored',
       'weights; restore every output row. One final estimator-call replacement',
       'plus inert helpers/proxy; all other parent bytes unchanged.','',
       'Exact network invariance follows from positive homogeneity in real',
       'arithmetic. Stored-weight rounding was independently checked; approximate',
       'estimator invariance is neither asserted nor required. The existing radial',
       'projection has a legitimate original-coordinate interpretation; this is',
       'an empirical coordinate choice, not a repair of a demonstrated bug.','',
       '| Rank | Lazy raw MSE | Normalized raw MSE | Raw ratio | Own-cold adjusted ratio |',
       '| --- | ---: | ---: | ---: | ---: |']
for r in score['rows']:
    lines.append(f"| {r['rank']} | {r['parent']['raw']:.16g} | {r['candidate']['raw']:.16g} | {r['raw_ratio']:.12g} | {r['adjusted_ratio']:.12g} |")
lines+=['',f"Pooled adjusted ratio **{score['pooled_adjusted_ratio']:.16g}**.",
        'The gate requires raw AND adjusted wins on BOTH networks and pooled<=.99.',
        'These are two reused ordinary1B development networks, not independent',
        'generalization evidence or an official score. All transformation and',
        'restoration work is included in the candidate complete cold bills.','']
for r in score['rows']:
    old,new=r['parent']['flops'],r['candidate']['flops']
    lines.append(f"Rank{r['rank']} cold bill {old}->{new} ({100*(new/old-1):+.6f}%). First-layer maximum absolute difference {r['first_layer_parent_max_abs']:.12g} (limit2e-5).")
lines+=['',
        'Three actual-FNP synthetic direct-network cases, zero-column handling,',
        'four malformed-input rejections, input preservation, exact repeated',
        'arrays/bills and column-norm checks pass.',
        f"Maximum scaled direct-network output error {max(max(r['layerwise_direct_output_scaled_errors']) for r in unit['cases']):.16g}.",
        'Four full calls pass own repeat, warm/cold100 bytes, restored first-layer',
        'tolerance, finite nonnegative16x1024 output and original cache retirement.',
        'Predictions and own bills were sealed before the two pinned truth files.','',
        f"Total full-call charge {resources['billed_prediction_flops']}F; peak RSS {peak}B.",
        'All four guarded stages exit0/retire with zero added swap. No numerical',
        'or upload worker remains. All eight prospective source files are unchanged.','']
if passed:
    lines+=['Only further fixed local qualification is earned. A wider panel and',
            'actual client/contract/package checks remain required before submission.','']
else:
    lines+=['Close this fixed normalization without a scale/target/layer/order/',
            'precision/kernel/head/metric/fit/blending search. No wider panel,',
            'production client, archive or submission follows. No universal claim',
            'about every positive scaling or jointly retrained estimator follows.','']
lines += [f"Official bank stays Lazy #331599 adjusted {bank['adjusted']:.16g}.",
          'The4e-9 milestone is achieved;3e-9,2e-9,~1e-9 remain active/unmet.',
          'Standing user authorization covers future qualified Phase2 uploads.','']
report='RESULT.md' if passed else 'NEGATIVE_RESULT.md'
with (HERE/report).open('x') as f:f.write('\n'.join(lines))
artifacts={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts}
size=sum((HERE/n).stat().st_size for n in artifacts)
assert size<15_000_000
closeout=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),status=status,
              goal_status='active',goal_achieved=False,official_bank=bank,
              candidate_sha256=build['candidate_sha256'],parent_sha256=build['parent_sha256'],
              source_inverse_byte_exact=True,unit_passed=True,development_gate_passed=passed,
              pooled_adjusted_ratio=score['pooled_adjusted_ratio'],
              raw_ratios=[r['raw_ratio'] for r in score['rows']],
              adjusted_ratios=[r['adjusted_ratio'] for r in score['rows']],
              first_layer_max_abs=[r['first_layer_parent_max_abs'] for r in score['rows']],
              resources=resources,artifact_bytes_before_closeout=size,
              artifacts_sha256=artifacts,remaining_disk_GiB=shutil.disk_usage(ROOT).free/2**30)
write_json(HERE/'closeout.json',closeout)
print(json.dumps({k:v for k,v in closeout.items() if k not in ['artifacts_sha256','official_bank']}))
