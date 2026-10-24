"""Stdlib-only sealing and checkpoint52; no numerical payload decoding."""
import datetime,json,os,shutil,subprocess
from pathlib import Path
from common import HERE,D,sha,pin,write,verify_seal
ROOT=D.parents[4]
def main():
    for name in ('sources_sealed.json','dependency_fix_sealed.json','capture_sealed.json'):verify_seal(name)
    old=json.loads((D/'continuation_checkpoint_v51_verified.json').read_text())
    for key in ('checkpoint','handoff','index','closeout'):assert sha(old[key]['path'])==old[key]['sha256'],key
    analysis=json.loads((HERE/'analysis.json').read_text());assert not analysis['passed']
    capture=json.loads((HERE/'capture/result.json').read_text());assert capture['completed'] and capture['repeat_all_captured_arrays_exact']
    assert json.loads((HERE/'unit.json').read_text())['passed']
    a,b=analysis['records'];assert (a['rank'],b['rank'])==(35,100)
    aa=a['layers'][-1]['free_mean'];cross_ratio=aa['full']['residual']['cross_mse']/aa['full']['target']['cross_mse']
    increment=aa['full']['residual']['cross_mse']/aa['mean_only']['residual']['cross_mse']
    assert cross_ratio==a['cross_gate']['full_parent_ratio'] and increment==a['cross_gate']['full_mean_only_ratio']
    assert cross_ratio>.75 and increment>.75
    br=b['ordinary_1B'];assert br['mse']['full']/br['parent_mse']==br['ratios']['full_parent']
    assert br['ratios']['full_parent']>.75 and br['ratios']['full_mean_only']>.75
    guards=[]
    for name in ('unit_watch.json','unit_v2_watch.json','capture_watch.json','analysis_watch.json'):
        g=json.loads((HERE/name).read_text());assert g['whole_group_retired'] and g['max_swap_growth_bytes']==0
        assert g['completed']==(name!='unit_watch.json')
        guards.append(dict(file=name,**g))
    assert 'ModuleNotFoundError: No module named' in (HERE/'unit.log').read_text()
    proc=subprocess.run(['ps','-Ao','pid,ppid,pgid,rss,etime,comm'],capture_output=True,text=True,check=True)
    others=[];unrelated=[]
    for line in proc.stdout.splitlines()[1:]:
        f=line.split(None,5)
        if len(f)!=6 or not any(s in f[5].lower() for s in ('python','whest','tensor')) or int(f[0])==os.getpid():continue
        cwdproc=subprocess.run(['/usr/sbin/lsof','-a','-p',f[0],'-d','cwd','-Fn'],capture_output=True,text=True)
        dirs=[x[1:] for x in cwdproc.stdout.splitlines() if x.startswith('n/')]
        record=dict(pid=int(f[0]),ppid=int(f[1]),pgid=int(f[2]),rss_KiB=int(f[3]),elapsed=f[4],executable=f[5],cwd=dirs)
        if not dirs:
            alive=subprocess.run(['ps','-p',f[0],'-o','pid='],capture_output=True,text=True)
            if alive.stdout.strip():others.append(record)
            else:record['status']='retired before cwd query';unrelated.append(record)
        elif any(Path(x).is_relative_to(ROOT) for x in dirs):others.append(record)
        else:unrelated.append(record)
    assert not others,others
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    peak=max([g['max_rss_bytes'] for g in guards]+[capture['peak_rss_bytes'],analysis['peak_rss_bytes'],json.loads((HERE/'unit.json').read_text())['peak_rss_bytes']])
    resources=dict(created_utc=now,groups_started=4,qualified_groups_completed=3,startup_dependency_failures=1,
        all_groups_retired=True,group_elapsed_s=sum(g['elapsed_s'] for g in guards),peak_rss_bytes=peak,
        max_swap_growth_bytes=0,estimator_calls=3,metered_parent_flops=sum(x['parent_flops'] for x in capture['records']),
        training_updates=0,new_samples=0,new_archive=False,new_upload=False,capture_bytes=capture['capture_total_bytes'],
        disk_free_GiB=shutil.disk_usage(D).free/2**30,current_other_matching_numerical_processes=others,unrelated_numerical_processes_observed=unrelated,process_scope="ARC workspace; unrelated jobs are observed but never stopped",guards=guards)
    write(HERE/'resources.json',resources)
    saved={}
    for key,backup in [('handoff','handoff_before_checkpoint52.md'),('index','index_before_checkpoint52.md')]:
        data=Path(old[key]['path']).read_bytes();saved[key]=data.decode()
        with (D/backup).open('xb') as f:f.write(data)
    source_seal=json.loads((HERE/'sources_sealed.json').read_text())
    external=[x for x in source_seal['pins'] if not Path(x['path']).is_relative_to(HERE)]
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
    source_pins=[pin(p) for p in sorted(HERE.glob('*.py'))]
    closeout=dict(completed=True,created_utc=now,status='Fixed post7 Gaussian-gain error transport gate failed; partial cross-replica signal retained',
        artifacts=artifacts,external_pins=external,own_source_pins=source_pins,artifact_bytes=sum(x['bytes'] for x in artifacts),
        new_accuracy_measurement=True,measurement_scope='privileged fixed transport diagnostic on two reused development networks',
        current_native_candidate=False,new_archive=False,new_upload=False,goal_achieved=False,
        next_numerical_study_selected=False,resources=resources,
        rank35_cross_gate=a['cross_gate'],ordinary_1B=[dict(rank=r['rank'],**r['ordinary_1B']) for r in analysis['records']])
    write(HERE/'closeout.json',closeout);cpin=pin(HERE/'closeout.json')
    heading=f'''## 2026-09-20 UTC — actual Lazy covariance-error transport measured; fixed halfway oracle gate failed

- Official Lazy #331599 remains adjusted3.8164492657795035e-9/raw2.169264529783277e-8.4e-9 bank achieved;3/2/~1e-9 remain unmet, goal active. Qualified Phase2 uploads remain preauthorized. No candidate, approval, owned numerical worker or upload pending.
- New study lazy_covariance_error_transport_v1 measured a real missing distinction: prior nonlinear covariance decoders used oracle previous m/C; older linear matrix observer learned from final mean only. Passive actual Lazy35/100/35 capture is ALL16/output/bill exact; all captured repeat arrays exact, W5 caches retired.379315752 capture bytes. Original source/archive untouched. Canonical weights and existing8M covariance source provenance verified, independent35 stream used (not cov/cov2 aliases).
- Fixed Gaussian-gain tangent initialized with true sampled post7 e/E, then NO later truth resets, versus same e7 with E7=0. Rank35 cross-replica final full/parent.817938763685 and full/mean-only.830563156336 FAIL both.75 gates. This is a partial18.2% noise-corrected point-estimate gain, not zero signal. Actual noisy forecast ordinary1B ratios1.130065620737/1.169457670333 on35/100; rank100 full/mean-only.987725704675 also fails. Not candidate adjusted scores or population/learnability ceilings; sampling/common MPS bias qualifications retained.
- Descriptive one-step rank35 covariance residual cross ratio.32463 at8 ->.19641 at14, final mean.13547; free post7-seeded covariance residual.84305 at14 and final mean.81794. Single-transition transport is useful, but source-free forecast loses accuracy over depth. No claim that fresh innovations are intrinsically unpredictable or that these teacher resets are deployable. No supplier established, fit/cutoff/strength/rank/feature rescue selected.
- Tiny tangent maximum4.34e-10, exact linearity8.37e-17. Three matched post7->pre8 covariance checks <=2.698e-5 (<1e-4). Captures sealed before targets; both forecasts sealed before either1B reference. One SciPy import startup failure preserved; dependency-only v2 uses math.erf with identical fixtures/gates. Three qualified groups complete, all four groups retired, zero added swap; peak RSS{peak}B, total guard wall{resources['group_elapsed_s']:.3f}s, three parent calls1164520978510F. No samples/training/archive/upload. About{resources['disk_free_GiB']:.2f}GiB free.
- RESULT.md/analysis.json/resources.json and closeout {cpin['sha256']} record the limited negative. An unrelated scheduled job outside ARC was observed and left untouched; worker-retirement statements refer to this study. Checkpoint52 supersedes51. All prior closures and checkpoint51's authenticated M/N/P stale-status correction remain valid. Next independent source/state/cost study is not selected; goal active, not blocked or achieved.

'''
    for key in ('handoff','index'):Path(old[key]['path']).write_text(heading+saved[key])
    cp=json.loads(Path(old['checkpoint']['path']).read_text())
    cp.update(updated_at_utc=now,goal_status='active',goal_achieved=False,checkpoint_version=52,
        supersedes='continuation_checkpoint_v51.json',current_handoff_sha256=sha(ROOT/'HANDOFF_CURRENT.md'),
        reexamination_index_sha256=sha(D/'REEXAMINATION_INDEX_20260919.md'),numerical_workers_active=False,
        approval_pending=False,pending_candidate=None,next_numerical_study_selected=False,
        previous_goal_turn_classification='source-status progress: completed older M/N/P fits authenticated; stale unlaunched label corrected',
        current_goal_turn_classification='numerical progress: privileged actual-Lazy covariance-error transport measured; fixed post7 Gaussian recurrence failed advancement gates')
    cp['lazy_covariance_error_transport']=dict(status=closeout['status'],closeout=cpin,
        rank35_cross_gate=a['cross_gate'],ordinary_1B_ratios={str(r['rank']):r['ordinary_1B']['ratios'] for r in analysis['records']},
        one_step_not_autonomous=True,not_universal_bound=True,resources=resources)
    write(D/'continuation_checkpoint_v52.json',cp)
    for row in artifacts+external+source_pins:assert sha(row['path'])==row['sha256'],row['path']
    verify=dict(checkpoint=pin(D/'continuation_checkpoint_v52.json'),handoff=pin(ROOT/'HANDOFF_CURRENT.md'),
        index=pin(D/'REEXAMINATION_INDEX_20260919.md'),closeout=cpin,all_study_pins_verified=True,
        scalar_gates_independently_replayed=True,numerical_workers_active=False)
    write(D/'continuation_checkpoint_v52_verified.json',verify);print(json.dumps(verify,indent=2))
if __name__=='__main__':main()
