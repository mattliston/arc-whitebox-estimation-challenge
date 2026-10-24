"""Fit the frozen seven-scale teacher repair from verified training predictions only."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='1'
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

HERE=Path(__file__).resolve().parent
RUNS=Path('/home/arcworker/arc-runs')
ROOT=Path('/home/arcworker/arc-workspace')
TRAIN=[35,150,200,250]
SHA_STUDENT='5b5f5f3e3e7609d788047732f3f81aa1ef3ad7449854f8f3163d84273e6e2a65'
SHA_TEACHER='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
SHA_GUARD='884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
SHA_CLIENT='1beba822cd02cebdfe9f8b30d0e08fb498a870a85fecaeccbe2fa38782098455'
SHA_SCHEDULER='315bc5227127b2fa7102fa937cb008edb7040c228527474826cd50a8d1f96aa2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def checked_train(tag, source_sha, evidence, fd_plan):
    folder=RUNS/tag
    run=read(folder/'RUN_MANIFEST.json');complete=read(folder/'COMPLETE.json')
    assert complete['successful'] and not complete['pending'] and all(v==0 for v in complete['results'].values())
    assert run['candidate_sha256']==source_sha and run['blas_threads']==4
    assert run['setup']=='paced' and run['warmup_each'] is False
    assert run['max_active']==36 and run['scheduler_sha256']==SHA_SCHEDULER
    assert complete['plan_sha256']==run['plan_sha256'] and complete['blas_threads']==4
    output={};receipt=dict(tag=tag,source_sha256=source_sha,
        manifest_sha256=sha(folder/'RUN_MANIFEST.json'),complete_sha256=sha(folder/'COMPLETE.json'),groups=[])
    for i,wanted in enumerate(([35,150,200],[250]),1):
        part=folder/f'part{i:02d}';gp=folder/f'part{i:02d}_GUARD.json';g=read(gp)
        assert g['guard_sha256']==SHA_GUARD and g['successful'] and g['whole_group_retired']
        assert g['child_exit']==0 and not g['cleanup_errors'] and g['reason'] is None
        assert g['max_owned_swap_bytes']==0 and g['cap_bytes']==6500000000 and g['reserve_bytes']==19000000000
        assert g['max_active']==36 and g['thread_provenance']['blas_threads']==4
        assert g['peak_cgroup_memory_bytes']<=g['cap_bytes'] and g['peak_group_rss_bytes']<=g['cap_bytes']
        assert not any(g['cgroup_events_final'][key] for key in ('oom','oom_kill','oom_group_kill','max'))
        plan_path=Path(g['command'][g['command'].index('--plan')+1]);plan=read(plan_path)
        assert sha(plan_path)==run['plan_sha256']
        ranks=plan['groups'][i-1]
        assert ranks==[35,150,200] if i==1 else ranks in ([250],[250,259])
        assert ranks[:len(wanted)]==wanted
        mf=read(part/'manifest.json');launch=read(part/'launcher.json')
        assert mf['sha256']==source_sha and mf['ranks']==ranks and mf['harness_sha256']==SHA_CLIENT
        assert mf['plan_sha256']==run['plan_sha256'] and mf['setup_mode']=='paced' and mf['warmup_each'] is False
        assert launch['error'] is None and launch['client_exit']==0 and launch['source_sha256']==source_sha
        assert sha(Path(mf['candidate']))==source_sha
        rows=[json.loads(x) for x in (part/'rows.jsonl').read_text().splitlines()]
        assert [x['rank'] for x in rows]==ranks
        entry=dict(guard_sha256=sha(gp),manifest_sha256=sha(part/'manifest.json'),
            rows_sha256=sha(part/'rows.jsonl'),plan_sha256=sha(plan_path),ranks=ranks,predictions=[])
        for row in rows[:len(wanted)]:
            rank=row['rank'];pp=part/f'pred_{rank}.bin'
            assert row['phase']=='scored' and row['nonfinite']==0 and row['n_out']==16384
            assert row['fast_rescue_stats'].get('calls',0)==0 and row['fast_rescue_control'].get('layer') is None
            assert mf['fixture_hashes'][str(rank)]==fd_plan['rows'][str(rank)]['weights']['sha256']
            assert sha(pp)==row['prediction_sha256'] and pp.stat().st_size==65536
            y=np.frombuffer(pp.read_bytes(),dtype='<f4').reshape(16,1024)[15].astype('f8')
            assert np.isfinite(y).all();output[rank]=y
            entry['predictions'].append(dict(rank=rank,prediction_sha256=sha(pp),flops=row['flops'],path=str(pp)))
        receipt['groups'].append(entry)
    assert sorted(output)==TRAIN
    evidence.append(receipt)
    return output

def solve(J,d):
    c=np.maximum(np.sqrt(np.mean(J*J,axis=0)),1e-12)
    A=J/c;gram=A.T@A/len(A)
    u=np.linalg.solve(gram+.1*np.eye(7),A.T@d/len(A))
    delta=u/c;scale=min(1.,.1/max(float(np.max(np.abs(delta))),1e-30))
    return delta*scale,dict(column_scales=c.tolist(),unclipped_delta=delta.tolist(),trust_multiplier=scale,
                           condition_regularized=float(np.linalg.cond(gram+.1*np.eye(7))))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    build=read(HERE/'BUILD_v1.json');plan=read(HERE/'PLAN_v1.json')
    assert sha(HERE/'PLAN_v1.json')==build['plan_sha256']=='52653db067995648869935a190ff7ee00381d3da5e1e4aea4c0723c9f17217a4'
    assert set(plan['rows'])==set(map(str,TRAIN)) and all(set(v)=={'weights'} for v in plan['rows'].values())
    while True:
        missing=[rec['name'] for rec in build['records'] if not (RUNS/rec['name']/'COMPLETE.json').exists()]
        if not missing:break
        if not args.watch:raise RuntimeError(('native FD still pending',missing))
        print(json.dumps(dict(utc=utc(),pending_arms=len(missing))),flush=True);time.sleep(20)
    evidence=[]
    teacher=checked_train('msg8_loaderfix337422_control32_markline_blas4_v1',SHA_TEACHER,evidence,plan)
    base=checked_train('msg8_edge_hidden6_nonlinear_blas4_v1',SHA_STUDENT,evidence,plan)
    arms={rec['name']:checked_train(rec['name'],rec['effective_sha256'],evidence,plan) for rec in build['records']}
    JJ=[];dd=[];curvature=[]
    for rank in TRAIN:
        norm=max(float(np.sqrt(np.mean(teacher[rank]**2))),1e-12)
        J=[]
        for layer in range(8,15):
            minus=arms[f'edge6_teacher_fd_l{layer}_minus_v1'][rank]
            plus=arms[f'edge6_teacher_fd_l{layer}_plus_v1'][rank]
            J.append((plus-minus)/.04/norm)
            curvature.append(dict(rank=rank,layer=layer,relative_second_difference=float(np.linalg.norm(plus+minus-2*base[rank])/max(np.linalg.norm(plus-minus),1e-20))))
        JJ.append(np.stack(J,axis=1));dd.append((teacher[rank]-base[rank])/norm)
    J=np.concatenate(JJ);d=np.concatenate(dd);delta,diagnostic=solve(J,d)
    loo=[]
    for i,rank in enumerate(TRAIN):
        step,diag=solve(np.concatenate([v for j,v in enumerate(JJ) if j!=i]),np.concatenate([v for j,v in enumerate(dd) if j!=i]))
        before=float(np.mean(dd[i]**2));after=float(np.mean((dd[i]-JJ[i]@step)**2))
        loo.append(dict(rank=rank,before_teacher_mse=before,linear_after_teacher_mse=after,ratio=after/before,delta=step.tolist()))
    before=float(np.mean(d*d));after=float(np.mean((d-J@delta)**2))
    # Export ONE fixed fit before any validation-three predictions are read.
    native=HERE/'native_v1';native.mkdir(exist_ok=False)
    base_dir=HERE.parent/'edge_hidden6_nonlinear_v1/native_hidden6_v1'
    science=base_dir/'candidate_msg8_edge_hidden6_nonlinear_science_v1.py'
    release=base_dir/'candidate_msg8_edge_hidden6_nonlinear_release_v1.py'
    assert sha(science)==SHA_STUDENT
    assert sha(release)=='51f8f466e2465f464ca0e1ce529a622229b2bdb3891cf4911c9efa99cf4ef928'
    tree=ast.parse(science.read_text());assignments={t.id:n.value for n in tree.body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
    old=ast.literal_eval(assignments['_GN_EDGE']);fitted={l:(1+old[l])*(1+float(delta[l-8]))-1 for l in range(8,15)}
    assert all(v!=0 for v in fitted.values()) and np.max(np.abs(delta))<=.10000000001
    tail='# Frozen seven-scale end-to-end teacher correction. Train4 only; no truth.\n_GN_EDGE = '+repr(fitted)+'\n_FAST_RESCUE_FRAC = {}\n'
    outputs={}
    for role,parent in [('science',science),('release',release)]:
        target=native/f'candidate_edge6_final_teacher_{role}_v1.py'
        with target.open('x') as f:f.write(parent.read_text()+'\n'+tail)
        outputs[role]=dict(path=str(target),sha256=sha(target),parent_sha256=sha(parent))
    result=dict(utc=utc(),script_sha256=sha(__file__),plan_sha256=sha(HERE/'PLAN_v1.json'),
        teacher_sha256=SHA_TEACHER,student_sha256=SHA_STUDENT,train=TRAIN,validation=[259,277,287],
        no_truth_read=True,no_validation_predictions_read=True,h=.02,ridge=.1,trust_bound=.1,
        delta=delta.tolist(),scales=(1+delta).tolist(),GN_EDGE=fitted,diagnostic=diagnostic,curvature=curvature,
        training_teacher_mse=before,linear_training_teacher_mse=after,linear_training_ratio=after/before,
        leave_one_network_out=loo,evidence=evidence,outputs=outputs,
        native_gain_unknown=True,no_native_validation_launched=True,no_auto_fire=True)
    write(HERE/'FITTED_v1.json',result)
    print(json.dumps({k:result[k] for k in ['utc','delta','linear_training_ratio','leave_one_network_out','outputs']},indent=2))

if __name__=='__main__':main()
