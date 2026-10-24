"""Independent flattened-dot reaggregation of all full-array observations."""
import math,resource
import numpy as np
from threadpoolctl import threadpool_info
from common import *
seal_check();done('measure');assert all(t['num_threads']==1 for t in threadpool_info());r=read(HERE/'result.json');meta=read(HERE/'inputs.json')
for p in (r['labels'],r['predictions'],r['inputs']):check(p)
ps=read(r['predictions']['path']);check(ps['predictions'])
with np.load(r['labels']['path'],allow_pickle=False) as z:labels={k:z[k] for k in z.files}
with np.load(ps['predictions']['path'],allow_pickle=False) as z:pred={k:z[k].astype(np.float64) for k in z.files}
with np.load(meta['capture']['path'],allow_pickle=False) as z:cap={k:z[k] for k in z.files if k.startswith(('sigma_','pdf_'))}
W=np.load(meta['weights']['path'],allow_pickle=False).astype(np.float64);mask=~np.eye(1024,dtype=bool)
def metric(a,b,d=None):
    a=a.ravel().astype(np.float64);b=b.ravel().astype(np.float64)
    n=len(a);ea=float(a@a/n);eb=float(b@b/n);observed=(ea+eb)/2;cross=float(a@b/n);v=a-b;dispersion=float(v@v/(2*n))
    ans=dict(energy_A=ea,energy_B=eb,observed_energy=observed,cross_energy=cross,dispersion_energy=dispersion,
        dispersion_fraction=dispersion/observed if observed else None,cosine=cross/math.sqrt(ea*eb) if ea and eb else None)
    return ans
maximum=0.;comparisons=0;independent_rows=[]
def close(a,b,scale):
    global maximum,comparisons
    if a is None or b is None:assert a is b;return
    err=abs(a-b)/max(abs(b),abs(scale),1e-30);assert err<=2e-10,(a,b,err)
    maximum=max(maximum,err);comparisons+=1
for row in r['layers']:
    l=row['layer'];a=labels[f'A_{l:02d}'];b=labels[f'B_{l:02d}'];wg=W[l+1]
    resp=cap[f'pdf_{l+1:02d}'].astype(np.float64)/(2*cap[f'sigma_{l+1:02d}'].astype(np.float64))
    def response(x):return np.einsum('ij,ij->j',wg,x@wg)*resp
    def channels(x):return dict(covariance=x,diagonal=np.diag(x),off_diagonal=x[mask],response=response(x))
    ac=channels(a);bc=channels(b);item=dict(baseline={},arms={})
    for key in ac:
        m=metric(ac[key],bc[key]);item['baseline'][key]=m
        for k,v in m.items():close(v,row['baseline'][key][k],m['observed_energy'] if k.endswith('energy') or k.startswith('energy') else 0)
        n=1024
        if key=='covariance':
            for k in ('energy_A','energy_B','observed_energy','cross_energy','dispersion_energy'):
                expected=(row['baseline']['diagonal'][k]*n+row['baseline']['off_diagonal'][k]*(n*n-n))/(n*n)
                close(m[k],expected,m['observed_energy'])
    for arm in ARMS:
        dc=channels(pred[f'{arm}_{l:02d}']);item['arms'][arm]={}
        for key in ac:
            m=metric(ac[key]-dc[key],bc[key]-dc[key]);base=item['baseline'][key]
            ga=base['energy_A']-m['energy_A'];gb=base['energy_B']-m['energy_B']
            delta=dc[key].ravel();gap=float(2*(delta@(ac[key]-bc[key]).ravel())/len(delta))
            m.update(ratio_A=m['energy_A']/base['energy_A'],ratio_B=m['energy_B']/base['energy_B'],cross_ratio=m['cross_energy']/base['cross_energy'],
                gain_A=ga,gain_B=gb,gain_gap=gap,correction_energy=float(delta@delta/len(delta)))
            close(ga-gb,gap,base['observed_energy'])
            for k,v in m.items():close(v,row['arms'][arm][key][k],base['observed_energy'] if k.endswith('energy') or k.startswith(('energy','gain')) else 0)
            item['arms'][arm][key]=m
    independent_rows.append(item)
for key,stored in r['pooled'].items():
    for arm in (None,*ARMS):
        items=[x['baseline'][key] if arm is None else x['arms'][arm][key] for x in independent_rows]
        expected=stored['baseline'] if arm is None else stored['arms'][arm]
        for k in ('energy_A','energy_B','observed_energy','cross_energy','dispersion_energy'):
            close(math.fsum(x[k] for x in items),expected[k],expected['observed_energy'])
        ea=math.fsum(x['energy_A'] for x in items);eb=math.fsum(x['energy_B'] for x in items)
        ob=(ea+eb)/2;disp=math.fsum(x['dispersion_energy'] for x in items);cr=math.fsum(x['cross_energy'] for x in items)
        close(disp/ob,expected['dispersion_fraction'],0);close(cr/math.sqrt(ea*eb),expected['cosine'],0)
        if arm:
            base=stored['baseline'];close(ea/base['energy_A'],expected['ratio_A'],0);close(eb/base['energy_B'],expected['ratio_B'],0)
            close(cr/base['cross_energy'],expected['cross_ratio'],0)
            for k in ('gain_A','gain_B','gain_gap','correction_energy'):close(math.fsum(x[k] for x in items),expected[k],base['observed_energy'])
attention=any(r['pooled'][k]['baseline']['dispersion_fraction']>=.20 for k in ('covariance','response'))
assert attention==r['further_label_acquisition_merits_separate_design'] and r['threshold']==.20
write(HERE/'audit.json',dict(passed=True,scalar_comparisons=comparisons,maximum_scaled_difference=maximum,
    full_array_dot_product_reaggregation=True,independent_fsum_pooling=True,all_gain_gap_identities=True,
    result=pin(HERE/'result.json'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,comparisons=comparisons,maximum_scaled_difference=maximum),flush=True)
