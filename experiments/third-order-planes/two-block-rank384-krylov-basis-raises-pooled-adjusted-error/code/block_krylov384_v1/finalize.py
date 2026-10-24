"""Close the fixed comparison, preserving the failed harness and all gates."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
D = HERE.parent
ROOT = Path('/Users/matt/arc-white-box-estimation')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')


seal = read(HERE/'sources_sealed.json')
for name,expected in seal['files'].items():
    assert sha(HERE/name)==expected,name
for name,expected in seal['external'].items():
    assert sha(name)==expected,name
seal2 = read(HERE/'sources_sealed_v2.json')
assert sha(HERE/'sources_sealed.json')==seal2['original_seal_sha256']
for name,expected in seal2['files'].items():
    assert sha(HERE/name)==expected,name
for row in read(HERE/'source_review.json')['sources']:
    assert sha(row['path'])==row['sha256'],row['path']
build = read(HERE/'build.json')
assert build['whole_source_inverse_byte_exact'] and build['last_push_basis_block_edits']==1
assert sha(D/'lazy_rotation_objective_v1/candidate.py')==build['parent_sha256']
unit = read(HERE/'unit_v2/result.json')
assert unit['passed'] and unit['repeated_arrays_byte_exact']
assert unit['repeated_fresh_module_bills_exact'] and unit['actual_inherited_kernel_used']
watches = [read(HERE/name) for name in
           ['unit_watch.json','unit_v2_watch.json','dev_watch.json','cold100_watch.json','score_watch.json']]
assert not watches[0]['completed'] and watches[0]['exit_code']==1
assert watches[0]['error'] is None
for w in watches[1:]:
    assert w['completed'] and w['exit_code']==0 and w['error'] is None
for w in watches:
    assert w['whole_group_retired'] and w['max_swap_growth_bytes']==0
    assert w['max_rss_bytes']<8_000_000_000
calls = []
for directory in ['dev','cold100']:
    result = read(HERE/directory/'result.json')
    assert result['candidate_sha256']==build['candidate_sha256']
    assert result['runner_sha256']==seal['external'][str(D/'weight5_reuse_v1/native.py')]
    assert not result['truth_accessed']
    for record in result['records']:
        assert sha(record['output'])==record['output_sha256']
        assert record['weight5_cache']['completed'] and record['weight5_cache']['numeric_cache_retired']
        assert record['multiplier']==max(.1,record['flops']/2**41)
        calls.append(record)
assert len(calls)==4
assert calls[0]['output_sha256']==calls[2]['output_sha256']
assert calls[1]['output_sha256']==calls[3]['output_sha256']
score = read(HERE/'dev_score.json')
assert score['repeat_byte_exact'] and score['warm_cold100_byte_exact']
assert score['first_five_layers_byte_exact'] and [r['rank'] for r in score['rows']]==[35,100]
assert not score['development_gate_passed']
peak = max([w['max_rss_bytes'] for w in watches]+[r['max_rss_bytes'] for r in calls]+[unit['max_rss_bytes']])
assert peak<8_000_000_000
resources = dict(full_prediction_calls=4,guarded_stages=5,
                 successful_guarded_stages=4,failed_harness_stages_preserved=1,
                 billed_prediction_flops=sum(r['flops'] for r in calls),
                 peak_rss_bytes=peak,additional_swap_bytes=0,all_owned_groups_retired=True,
                 total_guarded_wall_s=sum(w['elapsed_s'] for w in watches),
                 terminal_sessions=[13868,25887,78613,38968],active_session=None)
write_json(HERE/'resources.json',resources)
bank = read(D/'official_best_331599.json')
assert bank['submission_id']==331599
assert sha(D/'lazy_rotation_objective_v1/submission-lazy-objective-20260919.tar.gz')==bank['archive_sha256']
lines = ['# Fixed two-block Krylov384: closed negative','',
         'The192+192 basis retains two weighted-Gram Krylov blocks and a final',
         'joint QR at rank384. Only the last shared-push basis block and inert',
         'helpers changed; age4, original entry metric, physical factors, Lazy',
         'schedule, learned literals and all other parent bytes are unchanged.','',
         '| Rank | Lazy raw MSE | Krylov raw MSE | Raw ratio | Own-cold adjusted ratio |',
         '| --- | ---: | ---: | ---: | ---: |']
for row in score['rows']:
    lines.append(f"| {row['rank']} | {row['parent']['raw']:.16g} | {row['candidate']['raw']:.16g} | {row['raw_ratio']:.12g} | {row['adjusted_ratio']:.12g} |")
lines += ['',f"Pooled adjusted ratio: **{score['pooled_adjusted_ratio']:.16g}**.",
          'Both raw and adjusted scores worsen. The fixed both-wins and pooled',
          '<=.99 gate fails. These are two reused ordinary1B development',
          'networks, not independent generalization evidence or official scores.','']
for row in score['rows']:
    old,new = row['parent']['flops'],row['candidate']['flops']
    lines.append(f"Rank{row['rank']} complete cold FLOPs: {old} -> {new} ({100*(new/old-1):+.6f}%).")
lines += ['',
          'The leading operator multiply count alone understated the measured',
          'bill increase: block shapes, padding/kernel schedules, reorthogonalization,',
          'QR and changed network geometry all remain paid. No selective bill reuse.',
          '',
          'Four actual-kernel algebra cases pass against explicit FP64 dense Grams:',
          'joining-only, joining plus old core, an FP32-rounded rank32 construction,',
          'and an exactly zero operator. Rounding means the rank32 construction is',
          'not claimed to have exact algebraic rank32. Zero exercises exact degeneracy.',
          f"Maximum relative operator error: {max(r['operator_relative_error'] for r in unit['cases']):.16g}.",
          f"Maximum normalized orthogonality residual: {max(r['normalized_orthogonality'] for r in unit['cases']):.16g}.",
          'Krylov-block inclusion, input bytes, finite FP32 output, six malformed-input',
          'rejections, repeated arrays and fresh-module bills pass. The original',
          'unit harness stopped before a basis because its parent scratch arena',
          'was not initialized. That exit1 and source are preserved; additive',
          'unit_v2 fixes only arena setup/result location, with candidate and gates',
          'unchanged. See HARNESS_CORRECTION.md.','',
          'Four full predictions pass exact own repeat, warm/cold100 equality,',
          'first5 parent-layer bytes, finite nonnegative outputs and cache retirement.',
          'Own predictions and bills were sealed before opening pinned truth arrays.',
          f"Total full-call charge: {resources['billed_prediction_flops']} FLOPs.",
          f"Peak RSS including self-reports: {peak} bytes; zero added swap.",
          'All five groups retired: four successful stages and the preserved setup',
          'failure. No numerical or upload worker remains.','',
          'This construction is closed without rank/split/iteration/seed/layer/metric/',
          'QR/precision/head/blending/fit search. No wider panel, production client,',
          'package or submission follows this failed candidate. The result is not',
          'a general negative theorem about Krylov methods or alternative metrics.','',
          'General method background: [Musco and Musco2015](https://arxiv.org/abs/1504.05477).',
          'No approximation theorem from that paper is transferred to this fixed',
          'FP32 nonlinear estimator or its endpoint score.','',
          f"Official bank stays Lazy #331599 adjusted {bank['adjusted']:.16g}.",
          'The4e-9 milestone is achieved;3e-9,2e-9 and approximately1e-9 remain unmet.',
          'The goal stays active. Future qualified Phase2 uploads remain preauthorized.','']
with (HERE/'NEGATIVE_RESULT.md').open('x') as stream:
    stream.write('\n'.join(lines))
artifacts = {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts}
size = sum((HERE/name).stat().st_size for name in artifacts)
assert size<15_000_000
closeout = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                status='closed negative at fixed development gate',goal_status='active',
                goal_achieved=False,official_bank=bank,candidate_sha256=build['candidate_sha256'],
                parent_sha256=build['parent_sha256'],source_inverse_byte_exact=True,
                unit_v2_passed=True,unit_v1_setup_failure_preserved=True,
                development_gate_passed=False,pooled_adjusted_ratio=score['pooled_adjusted_ratio'],
                raw_ratios=[r['raw_ratio'] for r in score['rows']],
                adjusted_ratios=[r['adjusted_ratio'] for r in score['rows']],
                resources=resources,artifact_bytes_before_closeout=size,artifacts_sha256=artifacts,
                remaining_disk_GiB=shutil.disk_usage(ROOT).free/2**30)
write_json(HERE/'closeout.json',closeout)
print(json.dumps({k:v for k,v in closeout.items() if k not in ['artifacts_sha256','official_bank']}))
