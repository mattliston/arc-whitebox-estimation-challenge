"""Metadata-only closeout and checkpoint53 for the completed fixed fit."""
import datetime,json,os,shutil,subprocess
from pathlib import Path
from common import HERE,D,TRAIN,CHECK,read,write,sha,pin,verify
ROOT=D.parents[4]
def main():
    seals=['capture_sources_sealed.json','preparation_sources_sealed.json','acquisition_sealed.json','neural_sources_sealed.json','fit_sources_sealed.json']
    external={}
    for name in seals:
        s=verify(name)
        for r in s['pins']:
            if not Path(r['path']).is_relative_to(HERE):external[r['path']]=r
    old=read(D/'continuation_checkpoint_v52_verified.json')
    for k in ('checkpoint','handoff','index','closeout'):assert sha(old[k]['path'])==old[k]['sha256'],k
    result=read(HERE/'fit/result.json');assert result['completed'] and not result['passed'] and not result['check_targets_accessed']
    assert result['updates']==300 and result['backwards']==1200
    score=read(HERE/'fit/train_score.json');assert [r['rank'] for r in score['records']]==TRAIN
    raw=0.0
    for r in score['records']:raw+=r['raw']
    raw/=20
    assert raw==score['raw_mean'] and raw>2.5e-8
    gaussian=sum(r['gaussian_raw'] for r in score['records'])/20
    accesses=[json.loads(s) for s in (HERE/'fit/target_accesses.jsonl').read_text().splitlines()]
    assert len(accesses)==20 and {r['rank'] for r in accesses}==set(TRAIN) and all(r['role']=='TRAIN' for r in accesses)
    assert not (HERE/'fit/check_score.json').exists()
    pred=read(HERE/'fit/predictions_sealed.json');assert len(pred['records'])==24
    for r in pred['records']:assert sha(r['path'])==r['sha256'] and not r['forward_targets_provided']
    terminal=read(HERE/'fit/terminal_sealed.json');assert sha(terminal['path'])==terminal['sha256']==result['terminal_sha256']
    qual=read(HERE/'qualification/result.json');acq=read(HERE/'capture_all/result.json');neural=read(HERE/'neural_qualification/result.json');nr=read(HERE/'fnp_replay/result.json')
    assert all(x['passed'] for x in [qual,acq,neural,nr]) and len(neural['tiny_gradient'])==11
    guards=[]
    for name in ['qualification_watch.json','capture_all_watch.json','prepare_watch.json','fnp_replay_watch.json','neural_qualification_watch.json','fit_watch.json']:
        g=read(HERE/name);assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0;guards.append(dict(file=name,**g))
    proc=subprocess.run(['ps','-Ao','pid,ppid,pgid,rss,etime,comm'],capture_output=True,text=True,check=True)
    ours=[];unrelated=[]
    for line in proc.stdout.splitlines()[1:]:
        f=line.split(None,5)
        if len(f)!=6 or int(f[0])==os.getpid() or not any(s in f[5].lower() for s in ('python','whest','tensor')):continue
        q=subprocess.run(['/usr/sbin/lsof','-a','-p',f[0],'-d','cwd','-Fn'],capture_output=True,text=True)
        dirs=[s[1:] for s in q.stdout.splitlines() if s.startswith('n/')]
        record=dict(pid=int(f[0]),ppid=int(f[1]),pgid=int(f[2]),rss_KiB=int(f[3]),elapsed=f[4],executable=f[5],cwd=dirs)
        if not dirs:
            alive=subprocess.run(['ps','-p',f[0],'-o','pid='],capture_output=True,text=True)
            if alive.stdout.strip():ours.append(record)
            else:record['status']='retired during inventory';unrelated.append(record)
        elif any(Path(x).is_relative_to(ROOT) for x in dirs):ours.append(record)
        else:unrelated.append(record)
    assert not ours,ours
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    F=sum(r['host_feature_flops'] for r in qual['records']+acq['records'])+sum(r['flops'] for r in qual['controls'])+qual['tiny']['control_flops']+qual['tiny']['observed_flops']+sum(r['flops'] for r in nr['records'])
    peak=max([g['max_rss_bytes'] for g in guards]+[x['peak_rss_bytes'] for x in [qual,acq,neural,nr,result]])
    resources=dict(created_utc=now,groups_completed=6,all_groups_retired=True,total_guard_elapsed_s=sum(g['elapsed_s'] for g in guards),
        peak_observed_rss_bytes=peak,max_swap_growth_bytes=0,sampled_mps_driver_peak_bytes=max(neural['sampled_mps_driver_peak_bytes'],result['sampled_driver_bytes_max']),
        mps_driver_samples_are_not_transient_peak=True,full_native_gaussian_calls=28,tiny_native_gaussian_calls=2,neural_native_replays=4,
        metered_native_flops=F,training_epochs=60,optimizer_updates=300,per_network_backwards=1200,training_networks=20,check_mean_targets_decoded=0,
        new_physical_samples=0,new_archive=False,new_upload=False,current_owned_numerical_processes=ours,unrelated_processes_observed=unrelated,
        disk_free_GiB=shutil.disk_usage(D).free/2**30,guards=guards)
    write(HERE/'resources.json',resources)
    saved={}
    for key,backup in [('handoff','handoff_before_checkpoint53.md'),('index','index_before_checkpoint53.md')]:
        data=Path(old[key]['path']).read_bytes();saved[key]=data.decode()
        with (D/backup).open('xb') as f:f.write(data)
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
    source_pins=[pin(p) for p in sorted(HERE.glob('*.py'))]
    closeout=dict(completed=True,created_utc=now,status='Fixed Gaussian-only host plus autonomous final observer completed; TRAIN absolute accuracy gate failed',
        artifacts=artifacts,external_pins=list(external.values()),own_source_pins=source_pins,artifact_bytes=sum(r['bytes'] for r in artifacts),
        new_accuracy_measurement=True,new_native_fitted_candidate=False,new_archive=False,new_upload=False,goal_achieved=False,next_numerical_study_selected=False,
        final_train_raw=raw,gaussian_train_raw=gaussian,trained_to_gaussian_ratio=raw/gaussian,raw_gate=2.5e-8,
        provisional_floor_adjusted=score['adjusted'],provisional_ratio_to_lazy=score['ratio'],check_targets_accessed=False,
        terminal_payload=pin(terminal['path']),resources=resources)
    write(HERE/'closeout.json',closeout);cpin=pin(HERE/'closeout.json')
    heading=f'''## 2026-09-20 UTC — cheap Gaussian host plus autonomous observer trained; accuracy gate failed

- Official Lazy #331599 remains adjusted3.8164492657795035e-9/raw2.169264529783277e-8.4e-9 achieved;3/2/~1e-9 remain unmet, goal active. Qualified Phase2 uploads remain preauthorized. No candidate, approval, owned numerical worker or upload pending.
- New gaussian_frozen_observer_v1 is distinct from D-only plane distillation, old CP1792 tree students, Gaussian-shadow features on an expensive bank and the full-Lazy frozen observer. Pure stable Gaussian6 physical trajectory, no CP/star/radial/birth/skew suffix or learned feedback. Exact original30-feature schema with zero incoming higher moments, current64-state/48-message observer trained final-only. Warm accepted trunk, TRAIN-only normalizers, final head/bias/gain zero, fixed final rms.01. No claim the cheap Gaussian physical closure was previously accurate.
- Native physical/callback qualification35/100/35 and tiny exact outputs/ledgers/repeat passed. Host+features~66.6492GF, actual neural replay4.552455498GF <8GF allowance; below multiplier floor, NOT a complete fitted candidate bill. CPU/MPS/native nonzero/direction max2.38419e-7,11 tiny gradient checks max7.542e-8, CPU/MPS gradient1.01916e-6; all pass.24 feature rows plus qualification100 captured and sealed before mean targets.
- Full fixed60epochs/300steps/1200backwards complete; endpoint only. TRAIN20 Gaussian raw3.736508635518e-6 -> learned1.037641433545e-6 (ratio.277703475293,72.23% removal). Absolute gate<=2.5e-8 FAIL by41.51x. At floor.1, cached TRAIN adjusted1.037641433545e-7 versus matched Lazy3.907171531555e-9, ratio26.55735550,0/20 wins. Reused training data, not official or holdout score; no complete trained-native bill claimed. All24 predictions sealed; exactly20 TRAIN target accesses. CHECK means remain unopened, no CHECK score or endpoint port.
- Terminal payload ff359d821ff994a0eec039656315e8154e5ed9f055492d70370903e6854a3985 preserved. No fit extension/width/feature/normalizer/auxiliary-loss/physical-schedule rescue selected. Scoped negative, not a universal cheap-model impossibility. Previous post7 transport and all older closures stay closed in their original scopes.
- Six guards completed/retired,zero added swap; total guarded wall{resources['total_guard_elapsed_s']:.3f}s, observed RSS peak{peak}B, sampled MPS driver{resources['sampled_mps_driver_peak_bytes']}B (not transient peak).28 full Gaussian calls+2tiny+4neural replays total{F} meteredF; no new samples or dense weights copied. About{closeout['artifact_bytes']/1e6:.2f}MB artifacts and{resources['disk_free_GiB']:.2f}GiB disk free. RESULT.md/fit/result.json/resources.json and closeout{cpin['sha256']}. Checkpoint53 supersedes52; next independent study unselected, goal active and unachieved.

'''
    for key in ('handoff','index'):Path(old[key]['path']).write_text(heading+saved[key])
    cp=read(old['checkpoint']['path']);cp.update(updated_at_utc=now,goal_status='active',goal_achieved=False,checkpoint_version=53,
        supersedes='continuation_checkpoint_v52.json',current_handoff_sha256=sha(ROOT/'HANDOFF_CURRENT.md'),reexamination_index_sha256=sha(D/'REEXAMINATION_INDEX_20260919.md'),
        numerical_workers_active=False,approval_pending=False,pending_candidate=None,next_numerical_study_selected=False,
        previous_goal_turn_classification='numerical progress: actual-Lazy post7 covariance-error transport failed its fixed advancement gates',
        current_goal_turn_classification='numerical progress: qualified and trained cheap Gaussian-host final observer; fixed TRAIN accuracy gate failed')
    cp['gaussian_frozen_observer']=dict(status=closeout['status'],closeout=cpin,final_train_raw=raw,gaussian_train_raw=gaussian,
        provisional_ratio_to_lazy=score['ratio'],check_targets_accessed=False,terminal_payload=pin(terminal['path']),resources=resources)
    write(D/'continuation_checkpoint_v53.json',cp)
    for row in artifacts+list(external.values())+source_pins:assert sha(row['path'])==row['sha256'],row['path']
    v=dict(checkpoint=pin(D/'continuation_checkpoint_v53.json'),handoff=pin(ROOT/'HANDOFF_CURRENT.md'),index=pin(D/'REEXAMINATION_INDEX_20260919.md'),
        closeout=cpin,all_study_pins_verified=True,scalar_train_gate_and_target_roster_replayed=True,numerical_workers_active=False)
    write(D/'continuation_checkpoint_v53_verified.json',v);print(json.dumps(v,indent=2))
if __name__=='__main__':main()
