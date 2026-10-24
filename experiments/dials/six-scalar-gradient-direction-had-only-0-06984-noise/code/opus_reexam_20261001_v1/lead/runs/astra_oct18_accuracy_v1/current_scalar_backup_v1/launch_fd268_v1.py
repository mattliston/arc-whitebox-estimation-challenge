import json,os,subprocess,hashlib,math
from pathlib import Path
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
p=json.loads((D/'FD_RUN_PLAN_v1.json').read_text())
out=D/'fd268_v1';out.mkdir()
e=dict(os.environ,TRAINER_THREADS='1',TRAINER_PL='1',TRAINER_CARD='M',TRAINER_RELIEF='1',TRAINER_STAR_ATTACH='0',TRAINER_BIRTH_ATTACH='0')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):e[k]='1'
rows=[]
for arm in p['arms']:
    for path,h in p['sources'].items():assert sha(path)==h,path
    record=out/f'GUARD_{arm}_v1.json'
    command=[p['launcher'],'-B',p['guard'],'--record',str(record),'--seconds','180','--footprint-limit','7500000000','--rss-backstop','7500000000','/usr/bin/nice','-n','10',p['torch_python'],'-B',str(D/'run_scalar_fd_HELD_v1.py'),'--arm',arm,'--plan-sha256',p['plan_sha256'],'--w3module',p['w3module'],'--output',str(out/arm)]
    write(out/f'COMMAND_{arm}_v1.json',command)
    print('START',arm,flush=True)
    with (out/f'LOG_{arm}_v1.txt').open('x') as log:r=subprocess.run(command,cwd=p['repository'],env=e,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,(arm,r.returncode)
    g=json.loads(record.read_text());assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and not g['error'] and not g['remaining_rows'] and g['min_pressure_free_pct']>=50
    c=json.loads((out/arm/'COMPLETE_v1.json').read_text());assert c['arm']==arm and c['arrays_unchanged'] and c['fd_plan_sha256']==p['plan_sha256'] and sha(c['result'])==c['result_sha256']
    assert max(g['max_rss_bytes'],g['max_footprint_bytes'],c['peak_rss'])<7500000000
    assert math.isfinite(c['mse']);rows.append(dict(arm=arm,complete=c,guard=str(record),guard_sha256=sha(record)))
    print('RETIRED',arm,flush=True)
fd=json.loads((D/'FD_PLAN_v1.json').read_text()); baseline=json.loads((D/'gradient268_v1/GRADIENT_SUMMARY_v1.json').read_text())
lp,lm=[r['complete']['mse'] for r in rows]; eps=fd['epsilon']; numeric=(lp-lm)/(2*eps); ad=fd['autograd_slope_mse_units']; discrepancy=abs(numeric-ad)/abs(ad)
result=dict(status='FD_PASS' if numeric*ad>0 and discrepancy<=.25 else 'FD_FAILED',rows=rows,autograd_slope=ad,centered_fd_slope=numeric,fd_to_ad=numeric/ad,relative_slope_discrepancy=discrepancy,plus_mse=lp,minus_mse=lm,baseline_mse=baseline['mse'],even_curvature_loss=(lp+lm)/2-baseline['mse'],baseline_eval_grad_mode_caveat=True,parameters_updated=False,no_batch=True,no_optimizer=True,whole_group_retired=True)
write(out/'RESULT_v1.json',result)
print(json.dumps({k:result[k] for k in ('status','autograd_slope','centered_fd_slope','fd_to_ad','relative_slope_discrepancy','even_curvature_loss')}),flush=True)
