import numpy as np
from common import HERE,read,write,pin,verify,seal_check

seal_check();assert read(HERE/'unit.json')['passed']
build=read(HERE/'build.json');sealed=read(HERE/'predictions_sealed.json')
assert len(sealed['rows'])==15
packets={};rows=[]
for row in sealed['rows']:
    verify(row['receipt']);r=row['call'];p=pin(r['output']);assert p['sha256']==r['output_sha256']
    rec=read(row['receipt']['path'])
    source=next(x['candidate'] for x in build['variants'] if x['coordinate']==row['coordinate'] and x['label']==row['label'])
    assert rec['candidate_sha256']==source['sha256'] and r['includes_initialization']
    assert r['weight5_cache']['completed'] and r['weight5_cache']['numeric_cache_retired']
    y=np.load(r['output'],allow_pickle=False)
    assert y.shape==(16,1024) and y.dtype==np.float32 and np.isfinite(y).all() and (y>=0).all()
    control=next(x for x in build['controls'] if x['rank']==row['rank'] and x['coordinate']==row['coordinate'] and x['label']==row['label'])
    verify(control['output'])
    if row['label']=='base':
        old=np.load(control['output']['path'],allow_pickle=False)
        assert y[:5].tobytes()==old[:5].tobytes()
    key=(row['rank'],row['coordinate'],row['label'],row['repeat']);packets[key]=(r,y)
    rows.append(dict(rank=row['rank'],coordinate=row['coordinate'],label=row['label'],repeat=row['repeat'],
        flops=r['flops'],original_flops=control['flops'],cost_ratio=r['flops']/control['flops'],prediction=p))
repeat=packets[(36,0,'plus',True)];original=packets[(36,0,'plus',False)]
assert repeat[0]['flops']==original[0]['flops'] and repeat[1].tobytes()==original[1].tobytes()
old=read(build['old_response']['path']);cells=[]
for rank in [36,38]:
    base_y=packets[(rank,None,'base',False)][1];base=base_y[-1].astype(np.float64)
    for c in [0,8]:
        values={}
        for label in ['plus','minus','double']:
            y=packets[(rank,c,label,False)][1];assert y[:8].tobytes()==base_y[:8].tobytes()
            values[label]=y[-1]
        p,m,d=[values[k].astype(np.float64) for k in ['plus','minus','double']]
        center=(p-m)/2;forward=p-base
        spacing=np.sqrt(np.sum(np.spacing(np.abs(values['plus'])).astype(np.float64)**2)+np.sum(np.spacing(np.abs(values['minus'])).astype(np.float64)**2))
        snr=float(np.linalg.norm(p-m)/max(spacing,1e-30))
        asym=float(np.linalg.norm(forward-center)/max(np.linalg.norm(center),1e-30))
        curve=float(np.linalg.norm(d-2*p+base)/max(np.linalg.norm(forward),1e-30))
        previous=next(x for x in old['cells'] if x['rank']==rank and x['coordinate']==c)
        cells.append(dict(rank=rank,coordinate=c,centered_response_rms=float(np.sqrt(np.mean(center**2))),
            spacing_snr=snr,forward_vs_centered_relative=asym,double_curvature_relative=curve,
            original_asymmetry=previous['forward_vs_centered_relative'],original_curvature=previous['double_curvature_relative'],
            passed=snr>=20 and asym<=.10 and curve<=.10))
cost_passed=all(r['cost_ratio']<=1.20 for r in rows)
passed=cost_passed and all(c['passed'] for c in cells)
write(HERE/'response_result.json',dict(passed=passed,cost_passed=cost_passed,rows=rows,cells=cells,
      repeat_exact=True,base_first5_original_exact=True,probes_first8_own_base_exact=True,
      targets_opened=False,fit_steps=0,full_native_calls=15,full_native_flops=sum(r['flops'] for r in rows)))
write(HERE/'outcome.json',dict(passed=passed,stage='response',closed=not passed,
      next_supervised_design_licensed=passed,targets_opened=False,fitted=False,uploaded=False))
print(dict(passed=passed,cost_passed=cost_passed,cells=cells),flush=True)
