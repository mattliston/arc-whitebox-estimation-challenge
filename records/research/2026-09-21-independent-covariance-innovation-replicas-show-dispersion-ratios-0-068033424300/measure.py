import resource,time
import numpy as np
from threadpoolctl import threadpool_info
from common import *
from metrics import stats,arm_stats,pool
from tangent import tangent_cov
seal_check();done('predict');assert all(t['num_threads']==1 for t in threadpool_info());start=time.monotonic()
meta=read(HERE/'inputs.json');ps=read(HERE/'predictions_sealed.json');check(ps['predictions']);assert ps['no_new_fit']
with np.load(meta['capture']['path'],allow_pickle=False) as z:cap={k:z[k] for k in z.files}
W=np.load(meta['weights']['path'],allow_pickle=False).astype(np.float64)
refs={};affine={}
for stream in ('A','B'):
    with np.load(meta[f'truth_{stream}']['path'],allow_pickle=False) as z:
        assert int(z['P'])==8_000_000
        if 'rank' in z.files:assert int(z['rank'])==35
        tm=z['mc_post'].astype(np.float64);tc=z['mc_post_cov'].astype(np.float64)
        pm=z['mc_pre_m'][8].astype(np.float64);pc=z['mc_pre_cov'][8].astype(np.float64)
    mr=relative(W[8].T@tm[7],pm);cr=relative(W[8].T@tc[7]@W[8],pc);assert max(mr,cr)<=1e-4
    refs[stream]=(tm,tc);affine[stream]=dict(mean_relative=mr,covariance_relative=cr)
with np.load(meta['data']['path'],allow_pickle=False) as z:oldU=z['U']
with np.load(meta['old_statistics']['path'],allow_pickle=False) as z:oldstats={k:z[k] for k in ('cov_energy','value_energy')}
with np.load(ps['predictions']['path'],allow_pickle=False) as z:pred={k:z[k].astype(np.float64) for k in z.files}
sym=lambda x:.5*(x+x.T);mask=~np.eye(1024,dtype=bool);rows=[];labels={};checks=[]
for j,l in enumerate(range(8,15)):
    pre={k:cap[f'{k}_{l:02d}'].astype(np.float64) for k in ('sigma','alpha','pdf','gate','mg','C_pre')};U={}
    for stream,(tm,tc) in refs.items():
        ep=tm[l-1]-cap[f'm_{l-1:02d}'];Ep=sym(tc[l-1]-cap[f'C_post_{l-1:02d}'])
        U[stream]=sym(tc[l]-cap[f'C_post_{l:02d}'])-tangent_cov(pre,ep,Ep,W[l])
        assert np.isfinite(U[stream]).all() and np.array_equal(U[stream],U[stream].T)
        labels[f'{stream}_{l:02d}']=U[stream]
    assert U['A'].astype(np.float32).tobytes()==oldU[j].tobytes()
    tmA,tcA=refs['A'];tmB,tcB=refs['B']
    independent=sym(tcA[l]-tcB[l])-tangent_cov(pre,tmA[l-1]-tmB[l-1],sym(tcA[l-1]-tcB[l-1]),W[l])
    lin=relative(U['A']-U['B'],independent);assert lin<=2e-10,(l,lin)
    wg=W[l+1];response=cap[f'pdf_{l+1:02d}'].astype(np.float64)/(2*cap[f'sigma_{l+1:02d}'].astype(np.float64))
    def value(u):return np.sum(wg*(u@wg),axis=0)*response
    y={stream:value(u) for stream,u in U.items()}
    ce=float(np.mean(U['A']**2));ve=float(np.mean(y['A']**2))
    cem=abs(ce-oldstats['cov_energy'][j])/ce;vem=abs(ve-oldstats['value_energy'][j])/ve;assert max(cem,vem)<=2e-10
    verr=[]
    for stream in ('A','B'):
        reference=np.array([float(wg[:,i]@(U[stream]@wg[:,i]))*response[i] for i in np.arange(16)*67])
        er=relative(y[stream][np.arange(16)*67],reference);assert er<=2e-10;verr.append(er)
    channels={
        'covariance':(U['A'],U['B']),
        'diagonal':(np.diag(U['A']),np.diag(U['B'])),
        'off_diagonal':(U['A'][mask],U['B'][mask]),
        'response':(y['A'],y['B'])}
    baseline={key:stats(a,b) for key,(a,b) in channels.items()};arms={}
    for arm in ARMS:
        delta=pred[f'{arm}_{l:02d}'];dy=value(delta)
        deltas=dict(covariance=delta,diagonal=np.diag(delta),off_diagonal=delta[mask],response=dy)
        arms[arm]={key:arm_stats(a,b,deltas[key]) for key,(a,b) in channels.items()}
        for key in channels:assert abs(arms[arm][key]['dispersion_energy']-baseline[key]['dispersion_energy'])<=2e-10*max(baseline[key]['observed_energy'],1e-30)
    rows.append(dict(layer=l,baseline=baseline,arms=arms))
    checks.append(dict(layer=l,original_FP32_U_byte_exact=True,original_covariance_energy_relative=cem,
        original_response_energy_relative=vem,replica_linearity_relative=lin,independent_response_relative=max(verr)))
    print(dict(layer=l,covariance_dispersion_fraction=baseline['covariance']['dispersion_fraction'],response_dispersion_fraction=baseline['response']['dispersion_fraction']),flush=True)
pooled={key:dict(baseline=pool([r['baseline'][key] for r in rows]),arms={}) for key in channels}
for key in channels:
    base=pooled[key]['baseline']
    for arm in ARMS:
        items=[r['arms'][arm][key] for r in rows];a=pool(items)
        a.update(ratio_A=a['energy_A']/base['energy_A'],ratio_B=a['energy_B']/base['energy_B'],
            cross_ratio=a['cross_energy']/base['cross_energy'] if base['cross_energy'] else None,
            gain_A=sum(x['gain_A'] for x in items),gain_B=sum(x['gain_B'] for x in items),
            gain_gap=sum(x['gain_gap'] for x in items),correction_energy=sum(x['correction_energy'] for x in items))
        pooled[key]['arms'][arm]=a
path=HERE/'labels.npz'
with path.open('xb') as f:np.savez(f,**labels)
attention=any(pooled[k]['baseline']['dispersion_fraction']>=.20 for k in ('covariance','response'))
write(HERE/'result.json',dict(completed=True,rank=35,layers=rows,pooled=pooled,checks=checks,affine_checks=affine,
    labels=pin(path),predictions=pin(HERE/'predictions_sealed.json'),inputs=pin(HERE/'inputs.json'),
    further_label_acquisition_merits_separate_design=attention,threshold=.20,
    interpretation='two-stream dispersion point diagnostic on a frozen fitted parent; not independent numerical bias, confidence bound, held-out score or unbiased fitted-arm cross-risk',
    parent_and_corrections_have_stream_A_training_dependence=True,official_score=False,new_fit_steps=0,new_samples=0,new_native_calls=0,
    wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(completed=True,pooled=pooled,attention=attention),flush=True)
