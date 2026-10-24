"""Metadata-only closeout for the failed fixed TRAIN gate; no array decoding."""
import datetime,json,os,shutil,subprocess
from pathlib import Path
from common import HERE,D,TRAIN,read,write,pin,sha,verify
from seal_stage import guard
ROOT=D.parents[4]

def main():
    seals=[verify(),verify('learning_sources_sealed.json'),verify('captures_sealed.json'),verify('fit_inputs_sealed.json')]
    old=read(D/'continuation_checkpoint_v56_verified.json')
    for k in ['checkpoint','handoff','index','closeout']:assert sha(old[k]['path'])==old[k]['sha256']
    oldclose=read(old['closeout']['path'])
    for k in ['artifacts','external_pins','own_source_pins']:
        for x in oldclose[k]:assert sha(x['path'])==x['sha256'],x['path']
    fit=read(HERE/'fit_result.json');prep=read(HERE/'preparation.json')
    assert fit['completed'] and not fit['passed'] and fit['train_ratio']>.50
    assert fit['train_ranks']==TRAIN and fit['trainable_scalars']==3329 and fit['epochs']==100 and fit['updates']==2400
    assert fit['device']=='cpu' and fit['optimizer_and_leaves_retired'] and fit['reloaded_nontrainable']
    assert not fit['check_targets_accessed'] and not fit['check_forecasts_started']
    assert len(fit['epoch_log'])==100 and all(v['epoch']==i+1 and v['updates']==24*(i+1) for i,v in enumerate(fit['epoch_log']))
    assert abs(fit['train_mse']/fit['zero_mse']-fit['train_ratio'])<1e-15
    for k in ['terminal','schedule','source_predictions']:assert sha(fit[k]['path'])==fit[k]['sha256']
    assert prep['completed'] and prep['rows']==49152 and prep['source_cells']==48
    assert not prep['check_target_access'] and not prep['ordinary_1B_access']
    assert sha(prep['data']['path'])==prep['data']['sha256']
    assert len(prep['proofs'])==6
    for r in prep['proofs']:
        assert len(r['layers'])==8
        assert all(max(c['affine_mean_relative'],c['affine_covariance_relative'])<=1e-4 for c in r['layers'])
    units=[read(HERE/('unit_'+s+'.json')) for s in ['fnp','torch']]
    assert all(u['passed'] and not u['truth_accessed'] for u in units)
    assert all(r['scaled_error']<=2e-5 and r['repeat_exact'] for r in units[0]['rows'])
    assert units[1]['output_scaled_error']<=2e-5
    assert len(units[1]['parameter_gradient_checks'])==9
    assert max(c['relative'] for c in units[1]['parameter_gradient_checks'])<=2e-4
    for name in ['fit_terminal_sealed.json','forecasts','forecasts_sealed.json','score.json','candidate.py']:
        assert not (HERE/name).exists(),name
    captures=[];guards=[]
    for rank in [None]+TRAIN:
        name='qualification' if rank is None else str(rank)
        fn='capture_'+name+'_watch.json';guards.append(dict(file=fn,**guard(fn)))
        result=read(HERE/('capture_qualification/result.json' if rank is None else f'captures/rank{rank:04d}/result.json'))
        assert result['completed'] and not result['truth_accessed'] and result['source_proof']['inverse_byte_exact']
        if rank is None:assert result['repeat_all_capture_bytes_exact']
        for row in result['records']:
            assert row['all16_parent_bytes_and_bill_exact'] and row['numeric_cache_retired']
            if rank is None:assert row['old_physical_neural_common_fields_exact']
            assert sha(row['capture']['path'])==row['capture']['sha256']
            captures.append(row)
    for s in ['unit_fnp','unit_torch','prepare','fit']:
        fn=s+'_watch.json';guards.append(dict(file=fn,**guard(fn)))
    assert len(captures)==9 and len(guards)==11
    owned=[];outside=0
    processes=subprocess.run(['ps','-Ao','pid,ppid,pgid,rss,etime,comm'],capture_output=True,text=True,check=True)
    for line in processes.stdout.splitlines()[1:]:
        f=line.split(None,5)
        if len(f)!=6 or int(f[0])==os.getpid() or not any(s in f[5].lower() for s in ['python','whest','tensor']):continue
        q=subprocess.run(['/usr/sbin/lsof','-a','-p',f[0],'-d','cwd','-Fn'],capture_output=True,text=True)
        dirs=[s[1:] for s in q.stdout.splitlines() if s.startswith('n/')]
        if dirs and not any(Path(s).is_relative_to(ROOT) for s in dirs):outside+=1;continue
        live=subprocess.run(['ps','-p',f[0],'-o','pid='],capture_output=True,text=True)
        if live.stdout.strip():owned.append(dict(pid=int(f[0]),executable=f[5],cwd=dirs))
    assert not owned,owned
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    peak=max([g['max_rss_bytes'] for g in guards]+[x['peak_rss_bytes'] for x in captures+units+[prep,fit]])
    assert peak<8_000_000_000
    resource=dict(created_utc=now,groups_completed=11,all_groups_retired=True,
        total_guard_elapsed_s=sum(g['elapsed_s'] for g in guards),peak_observed_rss_bytes=peak,max_swap_growth_bytes=0,
        full_estimator_calls=9,all_full_calls_unchanged_parent=True,total_parent_capture_flops=sum(r['flops'] for r in captures),
        fnp_small_fixture_calls=6,small_fixture_flops=sum(sum(r['bills']) for r in units[0]['rows']),
        training_epochs=100,training_updates=2400,trained_scalars=3329,training_device='cpu',
        new_physical_samples=0,check_forecasts=0,check_target_arrays_accessed=False,native_candidate=False,
        new_archive=False,new_upload=False,current_owned_numerical_processes=owned,unrelated_numerical_processes_count=outside,
        disk_free_GiB=shutil.disk_usage(D).free/2**30,guards=guards)
    write(HERE/'resources.json',resource)
    reduction=100*(1-fit['train_ratio'])
    result=f'''# Completed negative: current-Lazy local mean-innovation predictor

The fixed source-learning screen failed. The terminal normalized TRAIN MSE
ratio was {fit['train_ratio']:.16f} against a required <=0.50: a {reduction:.6f}%
reduction from the zero-source predictor. All100 epochs and2400 optimizer
updates completed. This is a measured negative training result, not an
untrained proposal, an estimator score, or a capacity/information bound.
CHECK forecasts and CHECK target arrays were not opened in this study.

Motivation was checkpoint56's actual-Lazy error ledger: late mean innovations
have a material formal endpoint contribution. The new experiment learned
local mean defects after subtracting the fixed Gaussian-gain transport of
the measured incoming mean and covariance errors. It did not repeat the old
recurrent final-score observer objective. The ledger's oracle component size
did not establish that these102 causal features could predict the defects.

TRAIN was150,200,250,300,400,500; CHECK35,100. These are previously used public
research networks, with a network-disjoint fit rather than pristine holdouts.
Existing8M coherent TRAIN references supplied48 layer8..15 labels. All affine
mean/covariance provenance checks passed1e-4. No new Monte Carlo, fresh6 or
confirmation targets were accessed. Source labels include tangent mismatch,
omitted responses, parent approximation error and reference noise; they are
not a physical causal decomposition. Final labels also include served output
rounding/clipping.

One shared FP32 MLP102->32->1 (3329 scalars) used the current native30 features,
outgoing64 hidden values, and8 fixed layer indicators. Global input mean/std
and target RMS used TRAIN only. Fixed CPU seed2026092057, AdamW,100 seeded
permutations and24 batches/epoch were sealed before label decoding. The
terminal payload was saved, optimizer/training leaves retired, and a fresh
nontrainable payload reloaded for the reported MSE. No width, feature,
learning-rate, seed, epoch extension, final-loss refit or blend was attempted.
Terminal SHA256: {fit['terminal']['sha256']}.

Nine passive parent calls (qualification35/100/35 plus six cold TRAIN calls)
preserved every output byte across16 layers and each complete native bill.
Repeated captures and the35/100 common physical/neural fields match prior
LCET/LFTO records exactly. FNP source-model outputs matched NumPy exactly on
three small fixtures with repeatable bills. Torch output scaled error was
{units[1]['output_scaled_error']:.9g}; all9 fixed finite-difference parameter
gradient checks passed, maximum relative error
{max(c['relative'] for c in units[1]['parameter_gradient_checks']):.9g}.

The failed TRAIN gate closes this arm before forecasting or CHECK scoring.
No FNP estimator integration, complete candidate cost, NC14, confirmation,
client runtime, package or upload was licensed. The13 square-product tangent
deployment estimate remains circuit planning only; no adjusted gain is claimed.

All11 numerical groups retired; total guarded time
{resource['total_guard_elapsed_s']:.6f}s, peak observed RSS{peak:,} bytes,
zero added swap. Parent capture bills total{resource['total_parent_capture_flops']:,}
FLOPs; six tiny FNP fixture calls total{resource['small_fixture_flops']:,}.
Research label generation and CPU training have no production bill. No GPU.

Accepted Lazy#331599 remains official adjusted3.8164492657795035e-9,
raw2.169264529783277e-8. The4e-9 bank is achieved;3/2/~1e-9 are unmet.
The broader goal remains active and qualified future submissions are
preauthorized. No further numerical study is selected in this checkpoint.
'''
    with (HERE/'RESULT.md').open('x') as f:f.write(result)
    backups={};saved={}
    for k in ['handoff','index']:
        data=Path(old[k]['path']).read_bytes();saved[k]=data.decode();dest=D/(k+'_before_checkpoint57.md')
        with dest.open('xb') as f:f.write(data)
        backups[k]=pin(dest)
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
    total_bytes=sum(x['bytes'] for x in artifacts);assert total_bytes<2_000_000_000
    external={}
    for seal in seals:
        for x in seal['pins']:
            if not Path(x['path']).is_relative_to(HERE):
                if x['path'] in external:assert external[x['path']]==x
                external[x['path']]=x
    external=list(external.values());own=[pin(p) for p in sorted(HERE.glob('*.py'))]
    status='Completed100-epoch local mean-innovation fit; TRAIN MSE ratio .876624 fails .50; CHECK unopened'
    close=dict(completed=True,created_utc=now,status=status,artifacts=artifacts,external_pins=external,own_source_pins=own,
        artifact_bytes=total_bytes,fit=pin(HERE/'fit_result.json'),terminal=fit['terminal'],resources=resource,
        training_ratio=fit['train_ratio'],fixed_gate=.50,gate_passed=False,check_forecasts_started=False,
        check_target_arrays_accessed=False,new_native_candidate=False,new_archive=False,new_upload=False,
        goal_achieved=False,next_numerical_study_selected=False)
    write(HERE/'closeout.json',close);cpin=pin(HERE/'closeout.json')
    heading=f'''## 2026-09-20 UTC — local mean-innovation predictor completed; fixed TRAIN gate failed

- Official Lazy#331599 unchanged: adjusted3.8164492657795035e-9/raw2.169264529783277e-8.4e-9 bank achieved;3/2/~1e-9 unmet; goal active. Qualified uploads preauthorized. No candidate, approval, numerical worker or upload pending.
- New lazy_mean_innovation_observer_v1 learned measured local mean innovations after subtracting incoming true-error Gaussian-gain transport, distinct from old final-loss observer fits. TRAIN150/200/250/300/400/500, CHECK35/100 (reused public research networks, network-disjoint fit not pristine validation). Native30 features+H64+8 layer indicators feed shared FP32 MLP102->32->1,3329 scalars. Existing8M TRAIN labels only;48 affine provenance cells passed1e-4.
- Model/arithmetic/trainer/source construction fixed before label decoding. Three FNP fixtures exact to NumPy and repeatable; CPU Torch scaled error{units[1]['output_scaled_error']:.3g};9 finite-difference checks pass. All100 epochs/2400 updates completed with one frozen seed/schedule. Terminal TRAIN normalized MSE/zero={fit['train_ratio']:.15f} fails<=.50 ({reduction:.4f}% reduction). Optimizer/leaves retired and saved terminal restored nontrainable. This is a completed negative fit, not a capacity bound or estimator score. No rescue, extension or source/feature sweep.
- Nine original-parent captures passed all16 bytes/full bills;35/100/35 repeat and old common fields exact.11 groups retired,{resource['total_guard_elapsed_s']:.6f}s,peak{peak}B,zero added swap. Captures total{resource['total_parent_capture_flops']}F;6 small fixtures{resource['small_fixture_flops']}F. {total_bytes/1e9:.3f}GB artifacts preserved. Terminal{fit['terminal']['sha256']}; closeout{cpin['sha256']}.
- Failed TRAIN gate stops before CHECK forecasts/targets, native integration, NC14/confirmation/client/package/upload. No new MC/GPU/fresh targets. Checkpoint56 and all18 artifacts/37 external/5 source pins reverified. Checkpoint57 supersedes56; no new numerical study selected. Broad goal active/unachieved.

'''
    for k in ['handoff','index']:Path(old[k]['path']).write_text(heading+saved[k])
    cp=read(old['checkpoint']['path']);prior=cp['current_goal_turn_classification']
    classification='numerical progress: completed qualified current-Lazy local mean-innovation training; fixed TRAIN source-fit gate failed'
    cp.update(updated_at_utc=now,goal_status='active',goal_achieved=False,checkpoint_version=57,supersedes='continuation_checkpoint_v56.json',
        current_handoff_sha256=sha(ROOT/'HANDOFF_CURRENT.md'),reexamination_index_sha256=sha(D/'REEXAMINATION_INDEX_20260919.md'),source_review_index_sha256=sha(D/'REEXAMINATION_INDEX_20260919.md'),
        numerical_workers_active=False,upload_worker_active=False,active_numerical_session=None,approval_pending=False,pending_candidate=None,next_numerical_study_selected=False,
        prior_checkpoint='continuation_checkpoint_v56.json',prior_checkpoint_sha256=old['checkpoint']['sha256'],previous_checkpoint=old['checkpoint'],
        previous_goal_turn_classification=prior,current_goal_turn_classification=classification,this_goal_turn_classification=classification,
        this_turn_resources=resource,local_work='Completed fixed100-epoch local mean-innovation predictor. TRAIN ratio .876624 failed .50; stopped before CHECK forecasts/scoring. No native candidate or upload.',preserved_previous_handoff_and_index=backups)
    cp['lazy_mean_innovation_observer']=dict(status=status,closeout=cpin,train_ratio=fit['train_ratio'],gate=.50,resources=resource)
    write(D/'continuation_checkpoint_v57.json',cp)
    for x in artifacts+external+own:assert sha(x['path'])==x['sha256'],x['path']
    verified=dict(checkpoint=pin(D/'continuation_checkpoint_v57.json'),handoff=pin(ROOT/'HANDOFF_CURRENT.md'),index=pin(D/'REEXAMINATION_INDEX_20260919.md'),
        closeout=cpin,all_study_pins_verified=True,completed_training_and_failed_gate_verified=True,check_targets_unopened=True,numerical_workers_active=False)
    write(D/'continuation_checkpoint_v57_verified.json',verified);print(json.dumps(verified,indent=2))

if __name__=='__main__':main()
