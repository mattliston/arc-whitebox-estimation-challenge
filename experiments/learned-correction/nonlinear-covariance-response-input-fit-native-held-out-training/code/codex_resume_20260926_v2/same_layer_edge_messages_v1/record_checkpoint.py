"""Preserve the completed fixed pilot and update the actual root handoff."""
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent;ROOT=P2.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
protocol=read(HERE/'protocol.json');result=read(HERE/'assessment_matched.json')
assert result['matched_worker_order']
assert result['protocol_sha256']==sha(HERE/'protocol.json')
assert result['assessment_sha256']==sha(HERE/'assess.py')
out=H2/'continuation_checkpoint_v23.json';assert not out.exists()
names=['qualification','capture','native_zero','fit_message','fit_control','check_message','check_control','assessment_repair','matched_baseline','assessment_matched','transfer_native_message','transfer_native_control','fit_transfer']
guards={name:read(HERE/f'{name}_guard.json') for name in names}
assert all(g['completed'] and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0 for g in guards.values())
bank=read(H2/'official_bank_S31_v1.json')
assert sha(H2/'candidate_old_both_tail_trace_bounded_v6.py')==protocol['parent_sha256']
archive=ROOT/'submissions/submission-20260926-213059.tar.gz'
assert sha(archive)=='8d3ae68e1550ff801ff8688c64e93787fdab6d0fd1064fbf72e71c02b58db7ff'
message=result['message_vs_bank'];control=result['control_vs_bank'];contrast=result['message_vs_control']
cal=read(HERE/'calibration.json');zero=read(HERE/'native_zero.json');qual=read(HERE/'qualification.json')
transfer=read(HERE/'fit_transfer.json')
assert transfer['optimizer_steps']==0 and not transfer['check_or_valid_accessed']
message_transfer=[r for r in transfer['records'] if r['mode']=='message']
assert [r['rank'] for r in message_transfer]==[38,40]
assert len(cal['records'])==12 and all(r['all16_bytes_exact'] and r['bill_exact'] for r in cal['records'])
assert zero['passed'] and qual['passed']
sources=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p)
         and p.name not in ('progress.json','remainder_state.json')]
for mode in ('message','control'):
    for stage in ('zero','terminal'):
        b=read(HERE/f'build_{mode}_{stage}.json');p=Path(b['candidate']);assert sha(p)==b['sha256'];sources.append(p)
    sources.extend(p for p in (H2/f'same_layer_edge_{mode}_check8_v1').rglob('*') if p.is_file())
    sources.extend(p for p in (H2/f'same_layer_edge_{mode}_fit2_transfer_v1').rglob('*') if p.is_file())
sources.extend(p for p in (H2/'same_layer_edge_baseline_check8_v1').rglob('*') if p.is_file())
size=sum(p.stat().st_size for p in sources);assert size<100_000_000,size
now=datetime.now(timezone.utc);local=now.astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')
decision=result['decision']
failed_recording=read(HERE/'assessment_guard.json')
assert failed_recording['whole_group_retired'] and not failed_recording['remaining_rows'] and failed_recording['max_swap_growth_bytes']==0
resources=dict(max_rss_bytes=max(g['max_rss_bytes'] for g in guards.values()),swap_growth_bytes=0,
    disk_free_gib=shutil.disk_usage(ROOT).free/1024**3,new_artifact_bytes=size,all_groups_retired=True)
checkpoint=dict(asof_utc=now.isoformat(),goal='active;2e-9 and approximately1e-9 unmet',
    official_bank=bank,classification='Completed fixed covariance-input training pilot; no new banked score improvement.',
    user_steering='Board submissions remain authorized. Continue meaningful improvements without another permission request.',
    experiment=str(HERE.relative_to(ROOT)),protocol_sha256=sha(HERE/'protocol.json'),
    qualification=dict(zero_torch_max_abs=qual['zero_max_abs'],finite_nonzero_backward=True,
        native_zero=zero['records'],native_calibration_exact_networks=12),
    fit={mode:read(HERE/f'fit_{mode}'/'result.json') for mode in ('message','control')},
    assessment=result,fit_transfer_diagnostic=transfer,decision=decision,resources=resources,guards=guards,
    json_recording_repair=read(HERE/'assessment_json_repair_v1.json'),failed_recording_guard=read(HERE/'assessment_guard.json'),
    retained_training_component=read(H2/'thin_star_route_training_component_v1.json'),
    running_processes=[],pending_submissions=[],pending_approvals=[],valid40_accessed=False,
    supersedes='continuation_checkpoint_v22.json',
    evidence_index=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(sources))],
    next='Prospectively fix larger qualification before VALID/release.' if result['passes_pilot'] else
        'This pilot is closed. Do not tune an epoch, sign, amplitude or feature set using CHECK8. Neither the approximate training loss nor a feature-level metric is a banked gain. Thin-star remains below the submission threshold; the original score target remains open.')
with out.open('x') as fp:json.dump(checkpoint,fp,indent=2,allow_nan=False);fp.write('\n')
text=(f'\n\n## Checkpoint23 — {local}: nonlinear covariance response-input pilot\n\n'
    f'Official S31 #332517 remains **2.1736465056908115e-9 adjusted**, 100/100 complete and zero failures. '
    f'The bank source and submission archive hashes are unchanged. No new banked improvement.\n\n'
    f'Fixed FIT12/CHECK8, disjoint subsets of TRAIN160; VALID40 unopened. Two zero-initialized 12x16 matrices '
    f'were fitted for twelve updates in each arm, with all prior parameters frozen and a native residual offset. '
    f'The Torch card remains an approximate proposal model with a detached star basis. '
    f'Target-free checks passed: exact zero-addition outputs, finite backward, exact all16 native outputs in both '
    f'zero arms, and exact prediction/bill controls on all twelve calibration networks.\n\n'
    f'Native CHECK8 message arm: raw {message["raw_pct"]:+.6f}%, adjusted {message["adjusted_pct"]:+.6f}%, '
    f'{message["adjusted_wins"]}/8 wins, adjusted 95% CI {message["adjusted_ci95_pct"]}. '
    f'Local-feature control: adjusted {control["adjusted_pct"]:+.6f}%, {control["adjusted_wins"]}/8 wins. '
    f'Message versus control: {contrast["adjusted_pct"]:+.6f}% adjusted. **{decision}.** '
    f'All first eight output layers remain byte-exact to S31.\n\n'
    f'Baseline and both candidates use the same eight-network worker order. '
    f'The frozen FIT38/40 message diagnostic gives native raw changes '
    f'{message_transfer[0]["native_pct"]:+.5f}%/{message_transfer[1]["native_pct"]:+.5f}%; '
    f'the corresponding Torch prediction-change cosines are '
    f'{message_transfer[0]["delta_cosine"]:.3f}/{message_transfer[1]["delta_cosine"]:.3f}, '
    f'with relative response discrepancies {100*message_transfer[0]["delta_relative_l2_discrepancy"]:.1f}%/'
    f'{100*message_transfer[1]["delta_relative_l2_discrepancy"]:.1f}%. Thus some FIT gain transfers, '
    f'but CHECK generalization fails and the approximate forward response remains materially different. '
    f'This two-network diagnostic does not certify a derivative, alter the gates or select another model. '
    f'All thirteen successful guarded groups and one failed JSON-reporting attempt retired; maximum sampled RSS {resources["max_rss_bytes"]/1e9:.3f}GB, '
    f'zero added swap, {size/1e6:.3f}MB new artifacts, {resources["disk_free_gib"]:.1f}GiB free. '
    f'No approval, upload or grading is pending. See `same_layer_edge_messages_v1/assessment_matched.json`, '
    f'`same_layer_edge_messages_v1/RESEARCH_CONTEXT.md`, and `continuation_checkpoint_v23.json`.\n')
results=H2/'RESULTS.md';backup=H2/'RESULTS_before_checkpoint23.md'
assert not backup.exists();shutil.copy2(results,backup)
with results.open('a') as fp:fp.write(text)
handoff=ROOT/'HANDOFF_CURRENT.md';backup=ROOT/'HANDOFF_CURRENT.md.before_codex_resume_v2_checkpoint23'
assert handoff.exists() and not backup.exists();shutil.copy2(handoff,backup)
banner=(f'**{local} CODEX RESTART: CHECKPOINT23; NO NEW BANKED IMPROVEMENT.**\n'
    f'S31 #332517 remains2.1736465056908115e-9 adjusted,100/100 complete,zero failures; source5d439308 '
    f'and archive8d3ae68e unchanged. Submission authorization persists. No approval, upload, grading or numerical worker pending.\n\n'
    f'Fixed nonlinear covariance response-input pilot completed: TRAIN12 fit, disjoint TRAIN8 check, '
    f'no VALID access. Message adjusted{message["adjusted_pct"]:+.6f}%,{message["adjusted_wins"]}/8 wins; '
    f'equal-size local control adjusted{control["adjusted_pct"]:+.6f}%. {decision}. '
    f'No sign/amplitude/epoch rescue. Initial zero parity and target-free native calibration were exact. '
    f'Approximate Torch training is not a certified native gradient.\n\n'
    f'All workers and the repaired JSON-reporting attempt retired, peak{resources["max_rss_bytes"]/1e9:.3f}GB,zero added swap, '
    f'{resources["disk_free_gib"]:.1f}GiB free. Thin-star TRAIN8-0.57346% remains below the submission bar. '
    f'Restart: codex_resume_20260926_v2/continuation_checkpoint_v23.json and '
    f'same_layer_edge_messages_v1/assessment_matched.json. Goal active;2e-9 and approximately1e-9 unmet.\n\n')
handoff.write_text(banner+handoff.read_text())
with (H2/'NEXT_RESEARCH_v1.md').open('a') as fp:fp.write(text)
(HERE/'progress.json').write_text(json.dumps(dict(phase='fixed_pilot_complete',checkpoint=str(out),passes_pilot=result['passes_pilot'],running_processes=[],bank_unchanged=True),indent=2)+'\n')
print(json.dumps(dict(checkpoint=str(out),decision=decision,message_adjusted_pct=message['adjusted_pct'],control_adjusted_pct=control['adjusted_pct'],resources=resources),indent=2))
