"""Metadata-only closeout of a completed failed fixed native candidate."""
import datetime,json,shutil,subprocess
from pathlib import Path
from common import HERE,D,PARENT,PARENT_SHA,read,write,pin,sha,verify

def main():
    seal=verify();verify('predictions_sealed.json')
    old=read(D/'continuation_checkpoint_v59_verified.json')
    for k in ['checkpoint','handoff','index','closeout']:assert sha(old[k]['path'])==old[k]['sha256']
    prior=read(old['closeout']['path'])
    for k in ['artifacts','external_pins','own_source_pins']:
        for x in prior[k]:assert sha(x['path'])==x['sha256']
    assert sha(PARENT)==PARENT_SHA
    s=read(HERE/'score.json');u=read(HERE/'unit.json')
    assert s['completed'] and not s['gate_passed'] and u['passed']
    assert len(s['records'])==2 and s['fitted_parameters']==s['new_samples']==0
    calls=[]
    for name in ['qualification','candidate','cold100']:
        r=read(HERE/name/'result.json');assert r['completed'] and r['qualified']
        assert r['repeat_bytes_exact']==(name!='cold100') and not r['truth_accessed']
        for x in r['records']:
            assert x['first15_parent_bytes_exact'] and x['parent_bill_decomposition_exact'] and x['numeric_cache_retired']
            assert x['whole_flops']==x['parent_flops']+x['extra_feature_flops']+x['final_addition_flops']
            if name=='qualification':
                assert x['all16_parent_bytes_exact'] and x['final_addition_flops']==0
                assert max(x['details'][k] for k in ['reference_delta_relative','reference_rays_relative','reference_trace_relative','tdiag_relative'])<=.002
            else:assert x['final_addition_flops']==2048
            calls.append(dict(stage=name,**x))
    assert len(calls)==7
    guards=[]
    for name in ['unit','qualification','candidate','cold100','score']:
        g=read(HERE/(name+'_watch.json'))
        assert g['completed'] and g['exit_code']==0 and g['error'] is None and g['whole_group_retired']
        assert g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<8_000_000_000
        guards.append(dict(stage=name,**g))
    groups={g['pid'] for g in guards};owned=[]
    ps=subprocess.run(['/bin/ps','-axo','pid=,pgid=,comm='],text=True,capture_output=True,check=True)
    for line in ps.stdout.splitlines():
        f=line.split(None,2)
        if len(f)==3 and int(f[1]) in groups:owned.append(line)
    assert not owned,owned
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    peak=max([s['peak_rss_bytes'],u['peak_rss_bytes']]+[x['peak_rss_bytes'] for x in calls]+[g['max_rss_bytes'] for g in guards])
    resources=dict(created_utc=now,guard_groups=5,all_groups_completed_and_retired=True,total_guard_elapsed_s=sum(g['elapsed_s'] for g in guards),
        peak_observed_rss_bytes=peak,max_swap_growth_bytes=0,full_estimator_calls=7,enabled_candidate_calls=4,passive_qualification_calls=3,
        whole_prediction_flops=sum(x['whole_flops'] for x in calls),feature_flops=sum(x['extra_feature_flops'] for x in calls),
        synthetic_feature_calls=8,synthetic_flops=2*sum(x['flops'] for x in u['cases']),
        fitted_parameters=0,new_samples=0,new_candidate_experiment=True,new_archive=False,new_upload=False,
        current_owned_numerical_processes=owned,disk_free_GiB=shutil.disk_usage(D).free/2**30,guards=guards)
    write(HERE/'resources.json',resources)
    table='\n'.join(f"| {x['rank']} | {x['raw_ratio']:.9f} | {x['flops']/x['parent_flops']:.9f} | {x['adjusted_ratio']:.9f} |" for x in s['records'])
    text=f'''# Terminal third-cumulant/covariance sketch: qualified, negative native score

The fixed16-probe terminal T-times-C candidate completed qualification and
the actual native35/100/35 plus cold100 experiment. Its pooled adjusted
ratio is{s['pooled_adjusted_ratio']:.12f}; the predeclared requirement was
both individual ratios<1 and pooled<=.995. The candidate failed. No seed,
count, coefficient, layer window or learned payload was changed afterward.

| Rank | Raw ratio | Complete cold FLOP ratio | Adjusted ratio |
| --- | ---: | ---: | ---: |
{table}

The correction uses one incoming third-cumulant hyperedge and one covariance
edge, with all retained CP/star/trace owners. It reuses actual inherited
final CP pushes and replaces a first-leg contraction and radial trace with
one fixed16-sign sketch, seed2026092060. It adds the original fourth-order
marginal mean response after the unchanged final learned hook. This is an
approximate terminal readout, not a full fourth state or exact source reset.
It is distinct from the old Gaussian tree and Gaussian triangle sources.

Four width4 fixtures enumerate all signs and verify the dense cubic
contraction and quartic trace independently; maximum FP32 relative error
{max(x['relative'] for x in u['cases']):.9g}. The full35/100/35 passive
qualification preserves all16 original output bytes and exactly recovers
each parent bill after subtracting the paid feature. IndependentFP64
source replay differs by at most
{max(x['details']['reference_delta_relative'] for x in calls if x['stage']=='qualification'):.9g}
relative in the final correction. This qualifies finite arithmetic and
factor reuse, not the accuracy or variance of the chosen sketch.

The four enabled calls preserve all first15 outputs.35 repeat and warm/cold100
predictions and features are byte-identical. Each enabled bill is the exact
parent bill plus the measured producer and2048 FLOPs for the final addition.
All candidate predictions were sealed before opening the existing1B public
answers. These two networks are reused development cases, not new holdouts.
No panel/client/package gate is claimed and no archive was produced.

This turn also corrected a source-selection mistake before numerical launch:
the earlier GaussianR1 source already failed at .9935529312 and is not an
untested promising carrier. Its receipt, along with triangle and recentering
negatives, remains intact. The previous full-slice-energy fit remains closed.

Five groups completed and retired in{resources['total_guard_elapsed_s']:.6f}s,
peak{peak:,} bytes, zero added swap. Seven full calls totaling
{resources['whole_prediction_flops']:,} measured FLOPs, plus eight tiny
feature checks. No fit, new Monte Carlo, GPU, external message or upload.

Official Lazy331599 remains adjusted3.8164492657795035e-9/raw
2.169264529783277e-8.4e-9 achieved;3/2/~1e-9 unmet. The full goal remains
active and future qualified submissions remain preauthorized. This fixed
candidate is closed; no follow-on seed/count/amplitude rescue is selected.
'''
    with (HERE/'RESULT.md').open('x') as f:f.write(text)
    saved={};backups={}
    for k in ['handoff','index']:
        data=Path(old[k]['path']).read_bytes();saved[k]=data.decode();dest=D/(k+'_before_checkpoint60.md')
        with dest.open('xb') as f:f.write(data)
        backups[k]=pin(dest)
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
    external=[x for x in seal['pins'] if not Path(x['path']).is_relative_to(HERE)]
    own=[pin(p) for p in sorted(HERE.glob('*.py'))]
    status='Completed natural-strength terminal third-cumulant/covariance sketch; qualification passed, actual native adjusted objective failed'
    close=dict(completed=True,created_utc=now,status=status,artifacts=artifacts,external_pins=external,own_source_pins=own,
        artifact_bytes=sum(x['bytes'] for x in artifacts),result=pin(HERE/'score.json'),resources=resources,
        qualification_passed=True,development_gate_passed=False,pooled_adjusted_ratio=s['pooled_adjusted_ratio'],
        next_numerical_study_selected=False,new_archive=False,new_upload=False,goal_achieved=False)
    write(HERE/'closeout.json',close);cpin=pin(HERE/'closeout.json')
    ratios=[x['adjusted_ratio'] for x in s['records']]
    heading=f'''## 2026-09-20 UTC — terminal T-times-C sketch: qualified but native adjusted negative

- Official Lazy331599 remains adjusted3.8164492657795035e-9/raw2.169264529783277e-8;4e-9 achieved,3/2/~1e-9 unmet. Goal active; uploads preauthorized. No numerical worker, candidate approval or upload pending.
- Prior59 was completed numerical progress: full cubic slice-energy producer passes, fixed27-coefficient source fit CHECK.9728705444/.9934250029 fails.61 artifacts/44 external/10 own pins reverified. GaussianR1 initially appeared untested, but its actual .9935529312 same-source negative was found; no carrier/direction variant was run. Old triangle and raw-third recentering also already negative.
- New terminal_third_covariance_sketch_v1 is the one-T/one-C four-external-vertex source, all CP/star/trace owners, maskedFP32 responses, fixed16 Rademacher first-leg/trace sketch, seed2026092060, actual inherited pp15 reuse. Natural coefficient1, final mean addition after unchanged P15; not Gaussian tree, full fourth closure, odd-descendant carrier or fitted model.
- Four dense width4 complete-sign fixtures pass. Passive35/100/35 keeps all16 output bytes and exactly charges extra work; independentFP64 correction relative<=6.63e-5, repeat feature bytes exact. Enabled35/100/35+cold100 first15 bytes exact,35repeat and warm/cold100 final/features exact; own complete cost sealed before1B targets. Cold adjusted35/100 ratios{ratios}; pooled{s['pooled_adjusted_ratio']:.12f} fails both<1/pooled<=.995 combination. No seed/count/strength/window rescue. No broader panel/client/contract/package/upload.
- Five groups retired,{resources['total_guard_elapsed_s']:.6f}s,peak{peak}B,zero added swap. Seven full calls/{resources['whole_prediction_flops']}F plus eight tiny checks;0fit/newMC/GPU. Artifacts{close['artifact_bytes']/1e6:.3f}MB; closeout{cpin['sha256']}. Checkpoint60 supersedes59; next numerical study unselected, full goal active/unachieved.

'''
    for k in ['handoff','index']:Path(old[k]['path']).write_text(heading+saved[k])
    cp=read(old['checkpoint']['path']);previous=cp['current_goal_turn_classification']
    classification='numerical progress: qualified new terminal T-times-C sketch and completed natural-strength native objective negative'
    cp.update(updated_at_utc=now,goal_status='active',goal_achieved=False,checkpoint_version=60,supersedes='continuation_checkpoint_v59.json',
        current_handoff_sha256=sha(old['handoff']['path']),reexamination_index_sha256=sha(old['index']['path']),source_review_index_sha256=sha(old['index']['path']),
        numerical_workers_active=False,upload_worker_active=False,active_numerical_session=None,approval_pending=False,pending_candidate=None,next_numerical_study_selected=False,
        prior_checkpoint='continuation_checkpoint_v59.json',prior_checkpoint_sha256=old['checkpoint']['sha256'],previous_checkpoint=old['checkpoint'],
        previous_goal_turn_classification=previous,current_goal_turn_classification=classification,this_goal_turn_classification=classification,
        this_turn_resources=resources,local_work=status,preserved_previous_handoff_and_index=backups)
    cp['terminal_third_covariance_sketch']=dict(status=status,closeout=cpin,qualified=True,development_gate_passed=False,
        adjusted_ratios=ratios,pooled_adjusted_ratio=s['pooled_adjusted_ratio'],resources=resources)
    write(D/'continuation_checkpoint_v60.json',cp)
    for x in artifacts+external+own:assert sha(x['path'])==x['sha256'],x['path']
    verified=dict(checkpoint=pin(D/'continuation_checkpoint_v60.json'),handoff=pin(old['handoff']['path']),index=pin(old['index']['path']),
        closeout=cpin,all_study_pins_verified=True,parent_invariance_and_native_objective_verified=True,numerical_workers_active=False)
    write(D/'continuation_checkpoint_v60_verified.json',verified);print(json.dumps(verified,indent=2))
if __name__=='__main__':main()
