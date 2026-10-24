"""Independent scalar replay of the completed native optimization history."""
import ast,math
from pathlib import Path
import numpy as np
from common import HERE,read,write,pin,verify,seal_check,literal_source

seal_check();fit=read(HERE/'fit.json');assert fit['completed'] and fit['iterations']==36
schedule=read(HERE/'schedule.json');parents={r['rank']:r for r in read(HERE/'baselines_sealed.json')['rows']}
base=[float(x) for x in schedule['base_head']];theta=base.copy();centers=[];rows=[]
def loss(output,truth,flops):
    v=np.load(output,allow_pickle=False)[-1]
    mse=math.fsum((float(a)-float(b))**2 for a,b in zip(v,truth))/1024
    return mse*max(.1,flops/2**41)
truths={}
for rank,p in parents.items():
    verify(p['truth']);verify(p['output'])
    with np.load(p['truth']['path'],allow_pickle=False) as z:truths[rank]=z['mc_post'][-1].astype(np.float64)
baseline={r:loss(p['output']['path'],truths[r],p['flops']) for r,p in parents.items()}
max_loss_relative=0.;max_state_delta=0.
for t,item in enumerate(schedule['rows']):
    record=read(HERE/f'iteration_{t:02d}.json');assert record['schedule']==item and item['t']==t
    max_state_delta=max(max_state_delta,max(abs(a-b) for a,b in zip(theta,record['center_before'])))
    a=.1*(4.6/(t+4.6))**.602;c=(1/16)/(t+1)**.101;delta=item['delta'];rank=item['rank']
    assert item['a']==a and item['c']==c
    measured=[]
    for j,sign in enumerate([1,-1]):
        call=record['calls'][j];assert call['rank']==rank and call['includes_initialization']
        variant=record['variants'][j];verify(variant['source'])
        source=Path(variant['source']['path']).read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id=='_EDGE_LITERAL' for k in n.targets))
        head=np.asarray(ast.literal_eval(node.value)['output'],np.float32)
        actual=np.asarray([record['center_before'][k]+sign*c*delta[k] for k in range(16)],np.float32)
        assert head.tobytes()==actual.tobytes()
        y=loss(call['output'],truths[rank],call['flops']);expected=record['absolute_losses'][j]['adjusted']
        rel=abs(y-expected)/max(abs(y),1e-300);max_loss_relative=max(max_loss_relative,rel);assert rel<=1e-12
        measured.append(y/baseline[rank])
    lp,lm=measured;coefficient=(lp-lm)/(2*c)
    gradient=[coefficient*d for d in delta];norm=math.sqrt(math.fsum(g*g for g in gradient))
    trial=[theta[k]-a*gradient[k]/max(1.,norm) for k in range(16)]
    offset=[trial[k]-base[k] for k in range(16)];distance=math.sqrt(math.fsum(d*d for d in offset))
    scale=min(1.,.75/max(distance,1e-300));theta=[base[k]+offset[k]*scale for k in range(16)]
    change=max(abs(theta[k]-record['center'][k]) for k in range(16));max_state_delta=max(max_state_delta,change)
    assert change<=5e-12
    centers.append(theta.copy());rows.append(dict(iteration=t,independent_plus=lp,independent_minus=lm,maximum_center_delta=change))
head=np.asarray([math.fsum(row[j] for row in centers[-18:])/18 for j in range(16)],np.float32)
expected=np.asarray(fit['candidate']['head'],np.float32)
assert head.tobytes()==expected.tobytes();verify(fit['candidate']['source'])
write(HERE/'fit_audit.json',dict(passed=True,iterations=36,loss_evaluations=72,independent_scalar_loss_and_state_replay=True,
    maximum_loss_relative_error=max_loss_relative,maximum_center_absolute_delta=max_state_delta,
    terminal_fp32_head_exact=True,rows=rows,candidate=fit['candidate']['source'],new_samples=0,new_parameters_selected=False))
print(dict(passed=True,maximum_loss_relative_error=max_loss_relative,maximum_center_absolute_delta=max_state_delta,terminal_head_exact=True),flush=True)
