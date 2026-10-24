"""Preserve completed numerical evidence and the exact blocked upload action."""
import datetime
import shutil
from common import (HERE,D,CANDIDATE,CANDIDATE_SHA,PARENT_SHA,read,write,
                    sha,pin,verify,verify_pin)

verify()
for file in ('confirmation_sources_sealed.json','boundary_sources_sealed.json',
             'matched_runtime_sources_sealed.json','calibrated_release_sources_sealed.json',
             'scorer_repair_sealed.json'):
    for item in read(HERE/file)['pins']:
        verify_pin(item)
panel=read(HERE/'panel_score.json')
confirmation=read(HERE/'confirmation_score.json')
boundary=read(HERE/'boundary_comparison.json')
matched=read(HERE/'matched_runtime_comparison.json')
package=read(HERE/'package.json')
assert panel['panel_advancement_passed'] and confirmation['confirmation_advancement_passed']
assert boundary['native_byte_exact'] and boundary['native_flops_exact'] and boundary['repeat_byte_exact']
assert not boundary['boundary_gate_passed'] and matched['calibrated_runtime_gate_passed']
assert package['validated']
verify_pin(package['candidate']);verify_pin(package['archive'])
assert sha(CANDIDATE)==CANDIDATE_SHA
assert not (HERE/'submission.json').exists()
rejection=read(HERE/'upload_auto_review_rejection.json')
assert not rejection['process_created'] and not rejection['network_upload_occurred']

guards=[]
for path in sorted(HERE.rglob('*_watch.json')):
    guard=read(path)
    assert guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0,path
    if path.name=='score_watch.json':
        assert not guard['completed'] and guard['exit_code']==1
    else:
        assert guard['completed'] and guard['exit_code']==0,path
    guards.append(dict(record=pin(path),**guard))
full_calls=[]
for stage in ('panel','confirmation'):
    for row in read(HERE/stage/'completed.json')['records']:
        if not row.get('saved',False):
            full_calls.append(row['call'])
assert len(full_calls)==20
client_calls=[]
for stage in ('flms_panel_boundary','flms_parent_boundary'):
    rec=read(HERE/stage/'completed.json')
    assert rec['completed']
    client_calls.extend(rec['records'])
assert len(client_calls)==6
rss=max([g['max_rss_bytes'] for g in guards]+[c['max_rss_bytes'] for c in full_calls])
resources=dict(completed=True,all_owned_groups_retired=True,numerical_guards=len(guards),
    successful_guards=sum(g['completed'] for g in guards),preserved_harness_failures=1,
    peak_rss_bytes=rss,max_swap_growth_bytes=0,
    elapsed_guard_sum_s=sum(g['elapsed_s'] for g in guards),
    new_native_full_calls=20,new_client_full_calls=6,reused_development_cold_calls=2,
    new_full_estimator_flops=sum(c['flops'] for c in full_calls)+sum(c['summary']['flops_used'] for c in client_calls),
    contract_is_separate_tiny_shape=True,new_physical_samples=0,new_fits=0,new_gpu_jobs=0,
    artifacts_before_closeout_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    guards=[g['record'] for g in guards])
write(HERE/'resources.json',resources)

lines=['# Full lower-moment candidate confirmed; exact archive awaits upload approval','',
       'The preserved full-support candidate now has broader positive accuracy evidence.',
       'Its exact source was unchanged throughout this study. The official bank remains',
       'Lazy #331599 at adjusted3.8164492657795035e-9; no new upload occurred.',
       'The3e-9,2e-9 and approximately1e-9 objectives remain unmet.','',
       '| Comparison | Candidate adjusted | Parent adjusted | Adjusted ratio | Raw ratio | Wins |',
       '| --- | ---: | ---: | ---: | ---: | ---: |']
for name in ('public8','historical_fresh6','all14'):
    g=panel['groups'][name]
    lines.append(f"| {name}, noise-corrected4M | {g['candidate']['adjusted_nc']:.12g} | {g['parent']['adjusted_nc']:.12g} | {g['adjusted_nc_ratio']:.9f} | {g['raw_nc_ratio']:.9f} | {g['wins']} |")
lines.append(f"| Confirmation8, ordinary1B | {confirmation['aggregate']['candidate']['adjusted']:.12g} | {confirmation['aggregate']['parent']['adjusted']:.12g} | {confirmation['pooled_adjusted_ratio']:.9f} | {confirmation['pooled_raw_ratio']:.9f} | {confirmation['wins']} |")
lines += ['',
    'The twelve new panel rows alone have adjusted ratio'+str(panel['groups']['new_calls12']['adjusted_nc_ratio'])+'.',
    'All panel and confirmation parent scores reproduce exactly. Each candidate call',
    'owns its complete metered bill, including cold initialization. Panel prediction',
    'and cost files were sealed before reference decoding. These are reused research',
    'networks with existing references, not globally pristine validation or official scores.',
    '',
    'The original development5%-per-network gate remains failed. This study explicitly',
    'revised the research decision to investigate an already positive component; it',
    'did not change that old record or tune the candidate. Full covariance, coordinate',
    'star and trace support are a coupled intervention; FP32 summation order also',
    'differs, so this result is not an isolated causal attribution to one channel.',
    '',
    'The initial scorer stopped before opening references because it wrongly assumed',
    'two prefix rows were byte-identical on every network. score_v2 records per-layer',
    'comparisons instead. All first rows match; three second rows have tiny rounding',
    'differences. The failed scorer/source/seal/log are preserved; score formulas,',
    'candidate bytes, reference files, roster and advancement criteria were unchanged.',
    '',
    'Actual NumPy-free client35/100/35 output bytes and FLOP bills match native calls',
    'exactly, and repeated35 is exact. The original runtime gate FAILED: candidate',
    '47.0621/43.9091/44.8293s versus historical Lazy40.3462/36.8842/37.9341s.',
    'A separately predeclared contemporaneous control recovered the exact original',
    'Lazy outputs, bills and request counts, but took47.0112/42.8986/44.7678s.',
    f"Thus the parent slowed by{100*(matched['parent_current_to_old_wall_ratio']-1):.4f}%,",
    f"while candidate/current-parent total wall ratio is{matched['pooled_wall_ratio']:.12f}.",
    'The calibrated deployment gate passes, with fewer candidate requests on every',
    'call. The historical/absolute failure remains explicit. A board trial is still',
    'required to establish platform runtime and score; neither is guaranteed locally.',
    '',
    'Official whest contract and package validation pass. The779737-byte archive',
    'contains only estimator.py and manifest.json; their hashes and source bytes',
    'were verified inside the tarball.',
    f"Candidate SHA256:{CANDIDATE_SHA}",
    f"Archive SHA256:{package['archive']['sha256']}",
    '',
    'Automatic approval review rejected execution of submit_v2.py before creating',
    'a process or sending data: it requires trusted exact-payload authorization.',
    'A concrete approval request has been sent via request_user_input_async.',
    'The prior standing submission objective and preauthorization record remain',
    'preserved. Do not bypass the rejection. After explicit approval, submit this',
    'exact archive once, then poll the returned handle; do not regenerate a package',
    'or rerun completed numerical work merely because observation stops.',
    '',
    f"Resources:{resources['numerical_guards']} retired groups including one preserved scorer failure;",
    f"20 new native and6 client full calls,{resources['new_full_estimator_flops']}F total;",
    f"guarded wall sum{resources['elapsed_guard_sum_s']:.6f}s,peakRSS{rss}B,zero added swap.",
    'No new dataset, physical samples, GPU job or fit. Original bank and all previous',
    'sources/results are preserved. The goal remains active, not complete or blocked.','']
with (HERE/'RESULT.md').open('x') as stream:
    stream.write('\n'.join(lines))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
artifact_paths=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
external=[CANDIDATE,D/'lazy_rotation_objective_v1/candidate.py',
          D/'lazy_rotation_objective_v1/submission-lazy-objective-20260919.tar.gz',
          D/'full_lower_moment_objective_v1/closeout.json',D/'continuation_checkpoint_v60.json']
closeout=dict(completed=True,created_utc=now,
    status='Unchanged full lower-moment support confirmed on14+8; exact archive ready; upload auto-review requires user approval',
    artifacts=[pin(p) for p in artifact_paths],external_pins=[pin(p) for p in external],
    resources=resources,panel_advancement_passed=True,confirmation_advancement_passed=True,
    original_runtime_gate_passed=False,calibrated_runtime_gate_passed=True,
    package_validated=True,upload_attempt_created=False,upload_approval_pending=True,
    candidate_sha256=CANDIDATE_SHA,archive=package['archive'],
    official_bank=dict(id=331599,adjusted=3.8164492657795035e-9,raw=2.169264529783277e-8),
    goal_achieved=False,new_upload=False,result=pin(HERE/'RESULT.md'))
write(HERE/'closeout.json',closeout)
for item in closeout['artifacts']+closeout['external_pins']:
    verify_pin(item)
print(dict(status=closeout['status'],closeout=pin(HERE/'closeout.json'),resources=resources),flush=True)
