"""Metadata-only verification, closeout and checkpoint54."""
import datetime,os,shutil,subprocess
from pathlib import Path
from common import HERE,D,sha,pin,read,write,verify
ROOT=D.parents[4]
def main():
    seal=verify();old=read(D/'continuation_checkpoint_v53_verified.json')
    for key in ['checkpoint','handoff','index','closeout']:assert sha(old[key]['path'])==old[key]['sha256']
    r=read(HERE/'result.json');before=read(HERE/'before_1B_truth.json');unit=read(HERE/'unit.json')
    assert r['completed'] and not r['passed'] and not r['cross_gate'] and not r['ordinary_gate']
    assert unit['passed'] and len(unit['rows'])==12
    assert r['full_estimator_calls']==0 and r['native_final_replays']==9 and r['fitted_parameters']==0 and r['new_samples']==0
    assert before['all_predictions_fixed'] and not before['ordinary_1B_means_accessed']
    assert sha(before['predictions']['path'])==before['predictions']['sha256']==r['predictions']['sha256']
    for row in r['records']:
        assert row['basis_rank']==65 and row['basis_orthogonality']<=1e-10
        assert row['zero_maxabs']<=1e-6 and row['zero_rms']<=1e-7
        for stream in row['streams']:
            assert max(stream['affine_mean_relative'],stream['affine_covariance_relative'])<=1e-4
            assert stream['projection_residual_orthogonality']<=1e-10
        scores=row['ordinary_1B'];base=scores['parent_mse']
        for stream in scores['streams']:
            for arm in ['projected','full']:assert stream[arm]['mse']/base==stream[arm]['ratio']
            assert stream['projected']['ratio']>.80
    cross=r['records'][0]['cross_final']
    assert cross['parent']['cross_mse']>0
    for arm in ['projected','full']:assert cross[arm]['cross_mse']/cross['parent']['cross_mse']==cross[arm]['cross_ratio']
    assert cross['projected']['cross_ratio']>.75
    replay={x['label']:x for x in r['replays']}
    assert replay['35_0_projected']['output_sha256']==replay['35_0_projected_repeat']['output_sha256']
    assert replay['35_0_projected']['flops']==replay['35_0_projected_repeat']['flops']
    guards=[]
    for name in ['unit_watch.json','screen_watch.json']:
        g=read(HERE/name);assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
        guards.append(dict(file=name,**g))
    owned=[];outside_count=0
    proc=subprocess.run(['ps','-Ao','pid,ppid,pgid,rss,etime,comm'],capture_output=True,text=True,check=True)
    for line in proc.stdout.splitlines()[1:]:
        f=line.split(None,5)
        if len(f)!=6 or int(f[0])==os.getpid() or not any(s in f[5].lower() for s in ('python','whest','tensor')):continue
        q=subprocess.run(['/usr/sbin/lsof','-a','-p',f[0],'-d','cwd','-Fn'],capture_output=True,text=True)
        dirs=[s[1:] for s in q.stdout.splitlines() if s.startswith('n/')]
        if dirs and not any(Path(x).is_relative_to(ROOT) for x in dirs):outside_count+=1;continue
        alive=subprocess.run(['ps','-p',f[0],'-o','pid='],capture_output=True,text=True)
        if not alive.stdout.strip():continue
        owned.append(dict(pid=int(f[0]),executable=f[5],cwd=dirs))
    assert not owned,owned
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    F=r['setup_flops']+sum(x['flops'] for x in r['replays'])
    peak=max(r['peak_rss_bytes'],unit['peak_rss_bytes'],*(g['max_rss_bytes'] for g in guards))
    resources=dict(created_utc=now,groups_completed=2,all_groups_retired=True,total_guard_elapsed_s=sum(g['elapsed_s'] for g in guards),
        peak_observed_rss_bytes=peak,max_swap_growth_bytes=0,full_estimator_calls=0,native_final_replays=9,metered_final_replay_and_setup_flops=F,
        projection_and_truth_work_is_unmetered_research=True,fitted_parameters=0,new_physical_samples=0,new_archive=False,new_upload=False,
        current_owned_numerical_processes=owned,unrelated_numerical_processes_count=outside_count,disk_free_GiB=shutil.disk_usage(D).free/2**30,guards=guards)
    write(HERE/'resources.json',resources)
    result_text=f'''# Closed: terminal covariance-supervision prerequisite

The fixed covariance-space projection failed both advancement gates.
It captures substantial covariance Frobenius error but little final-variance
response error and does not improve the native final predictions.

| Reference | Projected MSE / parent | Full covariance control / parent |
| --- | ---: | ---: |
| 35, stream A | 1.0039280884164066 | 0.9507067444350218 |
| 35, stream B | 1.0017091930376703 | 0.9520061007263826 |
| 100, stream A | 1.0116283003206332 | 0.9009814335286453 |

These ordinary1B reference scores were computed after all nine native replay
outputs were fixed and sealed. Parent raw MSEs reproduce exactly. The required
projected ratio was <=.80 for every stream. Rank35 cross-stream final residual
ratio is 1.0062472828123523, failing <=.75; the descriptive full-control ratio
is .9284628947343043. This does not establish a population bound or rule out
coherent earlier joint mean/covariance changes.

The target-free active H14-plus-constant basis has rank65 on both networks.
Projected covariance residual energies are .551057/.557276 on35 and .506699
on100 (about44–49% removed), while Gaussian variance-response residual ratios
are .898546/.898493 and .979425 (about2–10% removed). Thus merely having many
covariance labels would not establish useful task information in this fixed
projection. Projection coefficients were privileged per network and chosen
only from covariance errors, not optimized for final means or shared/fitted.

All12 small projector fixtures pass. Actual basis orthogonality/projection
orthogonality pass; affine covariance provenance error is at most4.12409e-5,
under1e-4. Native zero replays pass their original tight output tolerance;
projected35-A repeats exactly in outputs and bill. All prevariances are
positive without strength scaling or repair. Inputs reuse exact accepted-Lazy
TCI captures and original/independent8M references, not new full trajectories.

The old oracle-trajectory P0 decoder, current final-mean bilinear fit and
layer7 free transport remain closed in their original scopes. This result
is a distinct finite covariance-supervision prerequisite on actual cumulative
error, not a new representation family or a successful rescue of those fits.
No covariance fit, rank/cutoff/support/scale variant, wider panel, package or
upload follows. The full reference control cannot advance for the failed
projected arm. No deployable supplier or complete adjusted bill is claimed.

Two guards completed and retired in {resources['total_guard_elapsed_s']:.6f}s;
peak observed RSS {peak:,} bytes, zero added swap. No full estimator calls,
new samples, training or GPU work. Nine native final-block replays plus setup
cost {F:,} metered FLOPs; covariance reference/projection work is research and
is not represented as a production bill. Disk free {resources['disk_free_GiB']:.2f}GiB.
Official Lazy#331599 remains adjusted3.8164492657795035e-9. Goal active;
qualified future Phase2 submissions remain preauthorized.
'''
    with (HERE/'RESULT.md').open('x') as f:f.write(result_text)
    saved={};backups={}
    for key,name in [('handoff','handoff_before_checkpoint54.md'),('index','index_before_checkpoint54.md')]:
        data=Path(old[key]['path']).read_bytes();saved[key]=data.decode()
        with (D/name).open('xb') as f:f.write(data)
        backups[key]=pin(D/name)
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
    assert sum(x['bytes'] for x in artifacts)<20_000_000
    external=[x for x in seal['pins'] if not Path(x['path']).is_relative_to(HERE)]
    own=[pin(p) for p in sorted(HERE.glob('*.py'))]
    closeout=dict(completed=True,created_utc=now,status='Fixed terminal covariance-supervision projection prerequisite failed',
        artifacts=artifacts,external_pins=external,own_source_pins=own,artifact_bytes=sum(x['bytes'] for x in artifacts),
        result=pin(HERE/'result.json'),projected_ordinary_ratios=[1.0039280884164066,1.0017091930376703,1.0116283003206332],
        projected_cross_ratio=cross['projected']['cross_ratio'],full_control_cross_ratio=cross['full']['cross_ratio'],
        resources=resources,new_native_candidate=False,new_archive=False,new_upload=False,goal_achieved=False,next_numerical_study_selected=False)
    write(HERE/'closeout.json',closeout);cpin=pin(HERE/'closeout.json')
    heading=f'''## 2026-09-20 UTC — covariance-supervision prerequisite: hidden-state projection has no final gain

- Official Lazy#331599 remains adjusted3.8164492657795035e-9/raw2.169264529783277e-8.4e-9 bank achieved;3/2/~1e-9 open, goal active. Qualified future uploads preauthorized; no candidate, approval, numerical worker or upload pending.
- Previous turn completed the cheap Gaussian observer fit and failed its TRAIN gate; checkpoint53 and prior artifact pins reverified. New terminal_covariance_supervision_v1 is related to old bilinear-history decoders but measures a distinct target/interface: covariance-space projection of actual Lazy cumulative post14 error, selected by covariance labels rather than final means. No old failure reclassified; no learned fit selected.
- Reused exact TCI35/100/35 tapes and original8M35/100 plus independent8M35. Target-free sqrt(Cdiag)*[H14,1] active basis, fixed RMS/SVD cutoff1e-7, rank65 both. Add orthogonal Frobenius projection of sym(Ctrue-C) to active covariance only, leave cold diagonal unchanged. Full-reference control substitutes reference active covariance and cold variances. Keep mu/CP/star/b/radial/H14/P fixed; replay complete native scalar/features/GNN. No joint-law coherence or deployable supplier claim.
- Projected ordinary1B MSE ratios35-A/B/100-A1.0039280884/1.0017091930/1.0116283003, all fail<=.80. Rank35 cross-residual ratio1.0062472828 fails<=.75. Covariance error removal44–49% yields only2–10% Gaussian next-variance response removal. Full-reference control improves ordinary raw4.8–9.9%, cross35 ratio.9284628947, descriptive only; cannot advance on behalf of projected arm. Parent scores exact, predictions sealed before1B references.
-12 projector fixtures, actual projection algebra, affine provenance<=4.12409e-5, zero native replay and projected35 repeat all pass.9 native final-block replays+setup{F} meteredF, zero full estimator calls/new samples/training. Two guards complete/retired{resources['total_guard_elapsed_s']:.6f}s,peak{peak}B,zero added swap. About{closeout['artifact_bytes']/1e6:.2f}MB artifacts. No rank/cutoff/support/scale/fit rescue, panel or upload. RESULT.md/result.json/resources.json and closeout{cpin['sha256']}; checkpoint54 supersedes53. Next independent study unselected; goal active and unachieved.

'''
    for key in ('handoff','index'):Path(old[key]['path']).write_text(heading+saved[key])
    cp=read(old['checkpoint']['path']);classification='numerical progress: fixed terminal covariance-supervision projection failed native final-error gates'
    cp.update(updated_at_utc=now,goal_status='active',goal_achieved=False,checkpoint_version=54,supersedes='continuation_checkpoint_v53.json',
        current_handoff_sha256=sha(ROOT/'HANDOFF_CURRENT.md'),reexamination_index_sha256=sha(D/'REEXAMINATION_INDEX_20260919.md'),
        source_review_index_sha256=sha(D/'REEXAMINATION_INDEX_20260919.md'),numerical_workers_active=False,upload_worker_active=False,
        active_numerical_session=None,approval_pending=False,pending_candidate=None,next_numerical_study_selected=False,
        prior_checkpoint='continuation_checkpoint_v53.json',prior_checkpoint_sha256=old['checkpoint']['sha256'],previous_checkpoint=old['checkpoint'],
        previous_goal_turn_classification='numerical progress: qualified/trained Gaussian-host observer failed fixed TRAIN accuracy gate',
        current_goal_turn_classification=classification,this_goal_turn_classification=classification,this_turn_resources=resources,
        local_work='Completed privileged current-Lazy terminal covariance projection and native final replays; fixed gates failed. No fit, candidate or upload; next study not selected.',
        preserved_previous_handoff_and_index=backups)
    cp['terminal_covariance_supervision']=dict(status=closeout['status'],closeout=cpin,projected_ordinary_ratios=closeout['projected_ordinary_ratios'],
        projected_cross_ratio=cross['projected']['cross_ratio'],full_control_cross_ratio=cross['full']['cross_ratio'],resources=resources)
    write(D/'continuation_checkpoint_v54.json',cp)
    for x in artifacts+external+own:assert sha(x['path'])==x['sha256'],x['path']
    verified=dict(checkpoint=pin(D/'continuation_checkpoint_v54.json'),handoff=pin(ROOT/'HANDOFF_CURRENT.md'),index=pin(D/'REEXAMINATION_INDEX_20260919.md'),
        closeout=cpin,all_study_pins_verified=True,scalar_gates_and_replay_roster_verified=True,numerical_workers_active=False)
    write(D/'continuation_checkpoint_v54_verified.json',verified)
    print(__import__('json').dumps(verified,indent=2))
if __name__=='__main__':main()
