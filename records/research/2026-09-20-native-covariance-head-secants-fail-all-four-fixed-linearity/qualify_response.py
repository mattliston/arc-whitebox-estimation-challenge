import numpy as np
from common import HERE,read,write,pin,verify,seal_check
seal_check();sealed=read(HERE/'qualification_predictions_sealed.json');inputs=read(HERE/'inputs.json');cells=[];rows=[]
for entry in sealed['rows']:
    verify(entry['source']);verify(entry['receipt']);verify(entry['watch'])
    call=entry['call'];p=np.load(call['output']);parent=next(x for x in inputs['qualification_parent'] if x['rank']==entry['rank']);verify(parent['output']);base=np.load(parent['output']['path'])
    assert p.shape==base.shape==(16,1024) and p.dtype==base.dtype==np.float32 and np.isfinite(p).all() and (p>=0).all()
    assert p[:8].tobytes()==base[:8].tobytes()
    rows.append(dict(rank=entry['rank'],coordinate=entry['coordinate'],label=entry['label'],repeat=entry['repeat'],first8_exact=True,
        flops=call['flops'],parent_flops=parent['flops'],cost_ratio=call['flops']/parent['flops'],output=pin(call['output'])))
for rank in [36,38]:
    parent=next(x for x in inputs['qualification_parent'] if x['rank']==rank);base=np.load(parent['output']['path'])[-1].astype(np.float64)
    for j in [0,8]:
        values={label:np.load(next(x for x in rows if x['rank']==rank and x['coordinate']==j and x['label']==label and not x['repeat'])['output']['path'])[-1] for label in ['plus','minus','double']}
        plus,minus,double=(values[k].astype(np.float64) for k in ['plus','minus','double'])
        centered=(plus-minus)/2;forward=plus-base
        scale=max(np.linalg.norm(centered),1e-30)
        spacing=np.sqrt(np.sum(np.spacing(np.abs(values['plus'])).astype(np.float64)**2)+np.sum(np.spacing(np.abs(values['minus'])).astype(np.float64)**2))
        snr=float(np.linalg.norm(plus-minus)/max(spacing,1e-30))
        asym=float(np.linalg.norm(forward-centered)/scale)
        nonlinear=float(np.linalg.norm(double-2*plus+base)/max(np.linalg.norm(forward),1e-30))
        cells.append(dict(rank=rank,coordinate=j,centered_response_rms=float(np.sqrt(np.mean(centered**2))),spacing_snr=snr,
            forward_vs_centered_relative=asym,double_curvature_relative=nonlinear,passed=snr>=20 and asym<=.10 and nonlinear<=.10))
a=next(x for x in rows if x['repeat']);b=next(x for x in rows if x['rank']==36 and x['coordinate']==0 and x['label']=='plus' and not x['repeat'])
repeat=np.load(a['output']['path']).tobytes()==np.load(b['output']['path']).tobytes() and a['flops']==b['flops'];assert repeat
q=dict(passed=all(x['passed'] for x in cells),cells=cells,rows=rows,repeat_exact=repeat,first8_exact_all_calls=True,
    full_native_calls=len(rows),full_native_flops=sum(x['flops'] for x in rows),predictions=pin(HERE/'qualification_predictions_sealed.json'),
    truth_arrays_accessed=False,fit_steps=0,qualification_scope='Only these sampled actual native secants; not a complete exact Jacobian or arbitrary finite-step claim.')
write(HERE/'response_qualified.json',q)
