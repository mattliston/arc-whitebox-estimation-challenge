"""Grade ORACLE3 after actual retirement. Pre-registered: mean MSE log change <= +0.15% passes."""
from pathlib import Path
import hashlib,json,math,subprocess
import numpy as np
D=Path(__file__).resolve().parent;S=D.parent/'opus_u64_screen_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((D/'PROTOCOL_REGISTERED_oracle_v1.json').read_text())
live={int(x) for x in subprocess.check_output(['/bin/ps','-axo','pid='],text=True).split()}
qr=json.loads((S/'SCREEN10_RESULT_v1.json').read_text());qrm={r['seed']:r['MSE_log_pct'] for r in qr['rows']}
rows=[]
for s in (0,1):
    h=D/f'run_oracle_screen_s{s}_v1';ready=json.loads((h/'READY_v1.json').read_text())
    assert ready['all_guards_retired'] and ready['controller_pid'] not in live
    for own in (json.loads(x) for x in (h/'ownership.jsonl').read_text().splitlines()):
        g=json.loads(Path(own['guard_record']).read_text());assert g['exit_code']==0 and g['whole_group_retired'] and own['owned_guard_pid'] not in live and g['pid'] not in live
    for e in ready['evaluations']:
        a=json.loads(Path(e['certificate']).read_text());ctl=next(x for x in c['stack_control_reuse'] if x['seed']==e['seed']);b=json.loads(Path(ctl['certificate']).read_text())
        ar,br=a['rows'][0],b['rows'][0];o=ar['stats']['_ASTRA_READ64_STATS']['oracle']
        lm=100*math.log(ar['mse_final']/br['mse_final'])
        rows.append(dict(seed=e['seed'],oracle_MSE_log_pct=lm,qr_tail_MSE_log_pct=qrm[e['seed']],oracle_layerMSE_log_pct=[100*math.log(x/y) for x,y in zip(ar['mse_layers'],br['mse_layers'])],
                         energy_dropped_svd=[round(x[2],6) for x in o],energy_dropped_qrtail=[round(x[3],6) for x in o],events=[(x[0],x[1]) for x in o]))
rows.sort(key=lambda r:r['seed'])
m=float(np.mean([r['oracle_MSE_log_pct'] for r in rows]))
out=dict(rule='mean oracle MSE log <= +0.15% passes',mean_oracle_MSE_log_pct=m,mean_qr_tail_MSE_log_pct=float(np.mean([r['qr_tail_MSE_log_pct'] for r in rows])),passed=m<=0.15,rows=rows,all_actual_retired=True)
p=D/'ORACLE3_RESULT_v1.json'
with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
for r in rows:print(r['seed'],'oracle %+.4f  qr %+.4f'%(r['oracle_MSE_log_pct'],r['qr_tail_MSE_log_pct']),'| svd-drop',r['energy_dropped_svd'],'| qr-drop',r['energy_dropped_qrtail'],'| L10-15 layerMSE',[round(x,3) for x in r['oracle_layerMSE_log_pct'][9:]])
print('MEAN oracle %+.4f  qr %+.4f  passed=%s'%(m,out['mean_qr_tail_MSE_log_pct'],out['passed']),'sha',sha(p)[:12])
