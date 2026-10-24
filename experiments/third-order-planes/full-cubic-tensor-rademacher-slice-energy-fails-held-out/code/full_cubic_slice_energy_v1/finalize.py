"""Metadata-only negative-study closeout and checkpoint59; no array decoding."""
import datetime,json,shutil,subprocess,os
from pathlib import Path
from common import HERE,D,PARENT,PARENT_SHA,TRAIN,read,write,pin,sha,verify

def main():
    seal=verify();learning=verify('learning_sealed.json');verify('unit_v2_sealed.json')
    old=read(D/'continuation_checkpoint_v58_verified.json')
    for k in ['checkpoint','handoff','index','closeout']:assert sha(old[k]['path'])==old[k]['sha256']
    prior=read(old['closeout']['path'])
    for k in ['artifacts','external_pins','own_source_pins']:
        for x in prior[k]:assert sha(x['path'])==x['sha256']
    assert sha(PARENT)==PARENT_SHA
    r=read(HERE/'information.json');u=read(HERE/'unit.json')
    assert r['completed'] and not r['qualified'] and u['passed']
    assert r['control_parameters']==24 and r['extended_parameters']==27
    assert not r['ordinary_1B_arrays_accessed'] and not r['native_candidate_created'] and not r['previous_failed_fit_reopened']
    q=read(HERE/'qualification/result.json');assert q['qualified'] and q['repeat_all_capture_bytes_exact']
    calls=q['records'][:]
    for rank in TRAIN:
        c=read(HERE/f'captures/rank{rank:04d}/result.json');assert c['qualified'] and not c['truth_accessed']
        calls+=c['records']
    assert len(calls)==9
    for c in calls:
        assert c['all16_parent_bytes_and_bill_exact'] and c['numeric_cache_retired'] and c['old_parent_feature_bytes_exact']
        assert c['whole_flops']-c['extra_flops']==c['parent_flops'] and c['extra_flop_ratio']<=.075
        assert c['extra_flops']==c['probe_setup_flops']+sum(x['flops'] for x in c['details'])
        for x in c['details']:assert x['z_relative']<=.001 and x['energy_relative']<=.002
    guards=[]
    names=['unit','unit_v2','qualification']+[f'train_{x}' for x in TRAIN]+['information']
    for name in names:
        g=read(HERE/(name+'_watch.json'))
        assert g['whole_group_retired'] and g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<8_000_000_000
        if name=='unit':assert not g['completed'] and g['exit_code']==1 and g['error'] is None
        else:assert g['completed'] and g['exit_code']==0 and g['error'] is None
        guards.append(dict(stage=name,**g))
    # Audit only groups launched by this study. Do not touch unrelated jobs.
    groups={g['pid'] for g in guards}
    ps=subprocess.run(['/bin/ps','-axo','pid=,pgid=,comm='],text=True,capture_output=True,check=True)
    owned=[]
    for line in ps.stdout.splitlines():
        f=line.split(None,2)
        if len(f)==3 and int(f[1]) in groups:owned.append(line)
    assert not owned,owned
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    peak=max([r['peak_rss_bytes'],u['peak_rss_bytes']]+[c['peak_rss_bytes'] for c in calls]+[g['max_rss_bytes'] for g in guards])
    resources=dict(created_utc=now,guard_groups=len(guards),groups_completed=len(guards)-1,preserved_fixture_failure=1,
        all_groups_retired=True,total_guard_elapsed_s=sum(g['elapsed_s'] for g in guards),peak_observed_rss_bytes=peak,max_swap_growth_bytes=0,
        full_estimator_calls=9,whole_estimator_flops=sum(c['whole_flops'] for c in calls),added_producer_flops=sum(c['extra_flops'] for c in calls),
        wide_feature_cells=72,wide_research_reference_cells=72,completed_synthetic_feature_calls=8,failed_fixture_feature_calls=2,
        closed_form_fits=4,control_parameters=24,extended_parameters=27,new_samples=0,new_candidate=False,new_archive=False,new_upload=False,
        ordinary_1B_arrays_accessed=False,current_owned_numerical_processes=owned,disk_free_GiB=shutil.disk_usage(D).free/2**30,guards=guards)
    write(HERE/'resources.json',resources)
    check=[x for x in r['records'] if x['split']=='CHECK'];pc=r['pooled']['CHECK'];pt=r['pooled']['TRAIN']
    table='\n'.join(f"| {x['rank']} | {x['extended_over_zero']:.9f} | {x['extended_over_control']:.9f} |" for x in check)
    text=f'''# Full cubic slice energy: producer qualified, information screen failed

The fixed16-pair descriptor contracts the entire explicitly rounded parent
cubic CP+star+trace tensor before squaring. All signed CP pairs and all
cross terms are included. This differs from the earlier top-eight covariance
mode feature, but it is still only one noisy scalar summary per coordinate.
It is not an exact Frobenius energy, a physical fourth cumulant closure, or
a source of privileged reference values at inference.

The producer is qualified on35/100/35 plus six cold TRAIN networks. All16
parent outputs and old feature bytes are exact, and whole minus added FLOPs
equals each original parent bill. Repeated feature bytes agree. Worst
contraction relative RMS is{max(x['z_relative'] for c in calls for x in c['details']):.9g};
worst energy error is{max(x['energy_relative'] for c in calls for x in c['details']):.9g}.
Added FLOPs range{min(c['extra_flop_ratio'] for c in calls):.6%} to
{max(c['extra_flop_ratio'] for c in calls):.6%}. These timings include research
FP64 references and are not client runtime qualification.

One fixed ridge fit predicts third/fourth cumulant errors using their
marginal mean-response weights. The extension has27 coefficients versus24
in the matched control. Six TRAIN networks150/200/250/300/400/500;
CHECK35/100 are disjoint from this fit but extensively reused public networks.
All layers8..15 participate. Four normal equations were solved once; no
hyperparameter, probe count/seed, or layer sweep was performed.

| CHECK rank | Loss / zero correction | Loss / fitted control |
| --- | ---: | ---: |
{table}

Pooled CHECK ratios are{pc['extended_over_zero']:.12f} and
{pc['extended_over_control']:.12f}, against required<=.85 and<=.95, plus
neither baseline may worsen on either network. Gates:{r['gates']}.
TRAIN ratios are{pt['extended_over_zero']:.12f} and
{pt['extended_over_control']:.12f}. The objective is the sum of separately
squared third/fourth mean-response errors, not final-layer error or an
official-score estimate. The finite8M references retain sampling noise.

The original unit fixture had A transposed in its explicit dense star tensor.
That failed log, source and guard are retained. Only the independent fixture
indexing was corrected in unit_v2; the producer and thresholds were unchanged.
Four corrected dense fixtures pass, including cold diagonals and signed-pair
cancellation. No network arrays were accessed by either unit attempt.

{len(guards)} groups retired ({len(guards)-1} successful and the preserved fixture
failure), {resources['total_guard_elapsed_s']:.6f}s total guard time, peak
{peak:,} bytes, zero added swap. Nine full parent calls,72 feature/reference
cells; no new MC, GPU,1B answers, native candidate, package, or upload.
The previous failed LMIO fit remains closed. This result closes this fixed
descriptor/regression screen, not every full-tensor supplier.

Official Lazy331599 remains adjusted3.8164492657795035e-9 and raw
2.169264529783277e-8. The4e-9 bank is achieved;3/2/~1e-9 are unmet. The
goal remains active and qualified future submissions remain preauthorized.
The next study must provide a more informative causal moment or off-diagonal
covariance source; no specific new numerical study is selected here.
'''
    with (HERE/'RESULT.md').open('x') as f:f.write(text)
    saved={};backups={}
    for k in ['handoff','index']:
        data=Path(old[k]['path']).read_bytes();saved[k]=data.decode();dest=D/(k+'_before_checkpoint59.md')
        with dest.open('xb') as f:f.write(data)
        backups[k]=pin(dest)
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
    external={x['path']:x for s in [seal,learning] for x in s['pins'] if not Path(x['path']).is_relative_to(HERE)}
    own=[pin(p) for p in sorted(HERE.glob('*.py'))]
    status='Completed full cubic slice-energy producer and fixed source regression; numerical/cost qualification passed, CHECK information gate failed'
    close=dict(completed=True,created_utc=now,status=status,artifacts=artifacts,external_pins=list(external.values()),own_source_pins=own,
        artifact_bytes=sum(x['bytes'] for x in artifacts),result=pin(HERE/'information.json'),resources=resources,
        producer_qualified=True,information_gate_passed=False,check_ratios=pc,train_ratios=pt,
        next_numerical_study_selected=False,new_candidate=False,new_archive=False,new_upload=False,goal_achieved=False)
    write(HERE/'closeout.json',close);cpin=pin(HERE/'closeout.json')
    heading=f'''## 2026-09-20 UTC — full cubic slice energy qualified but failed the fixed information gate

- Official Lazy331599 remains adjusted3.8164492657795035e-9/raw2.169264529783277e-8;4e-9 achieved,3/2/~1e-9 unmet. Goal active; qualified board uploads preauthorized. No candidate, approval, numerical worker or upload pending.
- New full_cubic_slice_energy_v1 computes16 fixed independent bilateral Rademacher contractions of complete rounded CP+star+trace tensor, then mean square, layers8..15. Paired center/difference algebra; all cross terms; full bank, not top8 modes. Fixed seed2026092059, no count/seed/layer sweep. Four tiny dense fixtures pass after preserving/correcting a transposed independent unit fixture; producer unchanged.
- Nine parent calls35/100/35 + TRAIN150/200/250/300/400/500: all16 outputs/F30/scalar bytes exact; whole minus actual addedF equals native parent bills; repeated feature bytes exact; all FP64 fidelity and7.5%cost gates pass. AddedF range{min(c['extra_flop_ratio'] for c in calls):.6%}..{max(c['extra_flop_ratio'] for c in calls):.6%}. Not a full estimator change or client/runtime qualification.
- Frozen single ridge1e-3 source fit:24 controls vs27 extended coefficients, actual standardized cumulant-error labels and parent marginal mean-response weights, all8 late layers. CHECK35/100 network-disjoint but reused research. Pooled extended/zero{pc['extended_over_zero']:.12f}, extended/control{pc['extended_over_control']:.12f}; fails protocol<=.85/.95 and each-network nonworsening combination. TRAIN{pt['extended_over_zero']:.12f}/{pt['extended_over_control']:.12f}. Full per-layer/network/component losses retained. Not final scores, causal closure, native gains or universal capacity bound. No LMIO fit reopened.
- {len(guards)} guards retired ({len(guards)-1} success+preserved fixture failure),{resources['total_guard_elapsed_s']:.6f}s,peak{peak}B,zero added swap. Nine calls/{resources['whole_estimator_flops']}F including{resources['added_producer_flops']} addedF;72 wide feature/reference cells; four linear fits. No new MC/GPU/1B arrays/native candidate/package/upload. Artifacts{close['artifact_bytes']/1e6:.3f}MB; closeout{cpin['sha256']}. Checkpoint59 supersedes58. No next numerical study selected; pursue more informative causal moment/offdiag covariance sources.

'''
    for k in ['handoff','index']:Path(old[k]['path']).write_text(heading+saved[k])
    cp=read(old['checkpoint']['path']);previous=cp['current_goal_turn_classification']
    classification='numerical progress: full cubic slice-energy producer qualified; completed fixed source regression failed CHECK information gate'
    cp.update(updated_at_utc=now,goal_status='active',goal_achieved=False,checkpoint_version=59,supersedes='continuation_checkpoint_v58.json',
        current_handoff_sha256=sha(old['handoff']['path']),reexamination_index_sha256=sha(old['index']['path']),source_review_index_sha256=sha(old['index']['path']),
        numerical_workers_active=False,upload_worker_active=False,active_numerical_session=None,approval_pending=False,pending_candidate=None,next_numerical_study_selected=False,
        prior_checkpoint='continuation_checkpoint_v58.json',prior_checkpoint_sha256=old['checkpoint']['sha256'],previous_checkpoint=old['checkpoint'],
        previous_goal_turn_classification=previous,current_goal_turn_classification=classification,this_goal_turn_classification=classification,
        this_turn_resources=resources,local_work=status,preserved_previous_handoff_and_index=backups)
    cp['full_cubic_slice_energy']=dict(status=status,closeout=cpin,producer_qualified=True,information_gate_passed=False,check_ratios=pc,train_ratios=pt,resources=resources)
    write(D/'continuation_checkpoint_v59.json',cp)
    for x in artifacts+list(external.values())+own:assert sha(x['path'])==x['sha256']
    verified=dict(checkpoint=pin(D/'continuation_checkpoint_v59.json'),handoff=pin(old['handoff']['path']),index=pin(old['index']['path']),closeout=cpin,
                  all_study_pins_verified=True,all_parent_output_and_bill_checks_verified=True,numerical_workers_active=False)
    write(D/'continuation_checkpoint_v59_verified.json',verified);print(json.dumps(verified,indent=2))
if __name__=='__main__':main()
