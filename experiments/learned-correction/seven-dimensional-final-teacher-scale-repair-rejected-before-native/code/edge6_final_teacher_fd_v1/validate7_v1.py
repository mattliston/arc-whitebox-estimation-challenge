"""Check the ONE frozen FD fit and run only train4/teacher-validation3."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/arcworker/arc-workspace');RUNS=Path('/home/arcworker/arc-runs')
PORT=HERE.parents[1]/'fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_markline_v1'
TAG='edge6_final_teacher_validation7_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');a=parser.parse_args()
    assert sys.platform=='linux' and HERE.is_relative_to(ROOT)
    f=read(HERE/'FITTED_v1.json');assert f['linear_training_ratio']<1
    assert f['no_truth_read'] and f['no_validation_predictions_read']
    assert f['train']==[35,150,200,250] and f['validation']==[259,277,287]
    for script,env,extra,out in [
        (HERE/'check_fitted_scale_v1.py','server',[],HERE/'native_v1/TINY_SCALE_v1.json'),
        (HERE.parent/'edge_hidden6_nonlinear_v1/check_import6_v1.py','client',
          ['--build',str(HERE/'FITTED_v1.json'),'--out',str(HERE/'native_v1/NUMPY_FREE_IMPORT_v1.json')],
          HERE/'native_v1/NUMPY_FREE_IMPORT_v1.json')]:
        if not out.exists():
            subprocess.run(['/home/arcworker/arc-envs/'+env+'/bin/python','-B',str(script)]+extra,check=True,timeout=60)
        r=read(out);assert r['successful'] and r['script_sha256']==sha(script)
    tiny=read(HERE/'native_v1/TINY_SCALE_v1.json');assert tiny['fitted_sha256']==sha(HERE/'FITTED_v1.json')
    imp=read(HERE/'native_v1/NUMPY_FREE_IMPORT_v1.json')
    assert {r['role']:r['sha256'] for r in imp['cases']}=={k:v['sha256'] for k,v in f['outputs'].items()}
    parent=PORT/'PLAN_MSG8_LOADERFIX_CONTROL_v1.json'
    pp=HERE/'PLAN_VALIDATION7_v1.json'
    if not pp.exists():
        plan=read(parent)
        plan.update(utc=utc(),kind='ONE_FIXED_EDGE6_FINAL_TEACHER_FIT_VALIDATION7_NO_TRUTH',
            groups=[[35,150,200],[250,259],[277,287]],first7=f['train']+f['validation'],held25=[])
        plan['rows']={str(r):{'weights':plan['rows'][str(r)]['weights']} for r in plan['first7']}
        sci=f['outputs']['science'];sp=Path(sci['path']);assert sha(sp)==sci['sha256']
        plan['candidates']={TAG:dict(path=str(sp.relative_to(ROOT)),sha256=sci['sha256'],effective_sha256=sci['sha256'])}
        plan['teacher_fit']=dict(fitted_sha256=sha(HERE/'FITTED_v1.json'),train=f['train'],validation=f['validation'],
            no_truth=True,one_frozen_candidate=True,validation_never_used_for_refit=True,
            rejection_rule='Stop if aggregate or majority val3 teacher fidelity worsens, training actual teacher MSE fails to improve, or strong curvature invalidates linear prediction.',
            no_auto_full32=True,no_auto_fire=True)
        write(pp,plan)
    plan=read(pp);assert plan['teacher_fit']['fitted_sha256']==sha(HERE/'FITTED_v1.json')
    pins=read(PORT/'BUILD_v1.json')
    for name,pin in pins['pins'].items():assert sha(PORT/name)==pin
    for name,pin in pins['original_v7_pins'].items():assert sha(PORT.parent/'cloud_linux_capacity_v7'/name)==pin
    if not a.run:print(json.dumps(dict(status='PREFLIGHT_PASS_NO_NATIVE',plan_sha256=sha(pp))));return
    assert not (HERE/'VALIDATION7_LAUNCH_v1.json').exists()
    assert not (RUNS/TAG).exists() and not (RUNS/(TAG+'_controller.log')).exists()
    cmd=['/home/arcworker/arc-envs/judge/bin/python','-B',str(PORT/'run_science_linux_markline_v1.py'),
        '--root',str(ROOT),'--plan',str(pp),'--candidate',TAG,'--out',str(RUNS/TAG),
        '--client-python','/home/arcworker/arc-envs/client/bin/python','--server-python','/home/arcworker/arc-envs/server/bin/python',
        '--state-dir',str(RUNS/'admission'),'--jobs','3','--max-active','36','--blas-threads','4',
        '--cgroup-root','/sys/fs/cgroup/arc-estimator-v5','--setup','paced']
    with (RUNS/(TAG+'_controller.log')).open('x') as stream:
        child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    r=dict(utc=utc(),pid=child.pid,tag=TAG,command=cmd,plan_sha256=sha(pp),source_sha256=f['outputs']['science']['sha256'],
        fitted_sha256=sha(HERE/'FITTED_v1.json'),tiny_sha256=sha(HERE/'native_v1/TINY_SCALE_v1.json'),
        import_sha256=sha(HERE/'native_v1/NUMPY_FREE_IMPORT_v1.json'),no_truth=True,only7=True,
        guard_unchanged=True,no_auto_full32=True,no_auto_fire=True,script_sha256=sha(__file__))
    write(HERE/'VALIDATION7_LAUNCH_v1.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()

