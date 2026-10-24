import resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import HERE,read,write,pin,verify,seal_check
from observer_math import raw_features,raw_design,direct,normalize_design,fnp_step

seal_check('fit_sources_sealed.json');assert read(HERE/'unit.json')['passed'] and read(HERE/'captures_sealed.json')['passed']
(HERE/'design').mkdir(exist_ok=False)
train=read(HERE/'captures_sealed.json')['rows'];check=[dict(rank=35,record=read(HERE/'capture_triple/result.json')['records'][0]),dict(rank=100,record=read(HERE/'capture_cold100/result.json')['records'][0])]
records=[];wide=[]
for row in check+train:
    c=row['record'];verify(c['capture']);verify(c['weights'])
    with np.load(c['capture']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    raw=raw_features(a);sigma=a['sigma'][8:];gate=a['features'][8:,:,5].astype(np.float64)
    w=np.load(c['weights']['path'],mmap_mode='r',allow_pickle=False)[8:]
    b=raw_design(raw,sigma,gate,w);assert b.shape==(8,1024,102) and np.isfinite(b).all()
    path=HERE/'design'/f'rank{row["rank"]:04d}.npy';np.save(path,b)
    if row['rank'] in [35,100]:
        mean=np.zeros(94);scale=np.ones(94);beta=1e-6*np.sin(.13*np.arange(1,103))
        expected=direct(raw,sigma,gate,w,mean,scale,beta)
        relative=float(np.max(np.abs(b@beta-expected))/max(float(np.max(np.abs(expected))),1e-12));assert relative<=2e-10
        cells=[]
        for zero in [False,True]:
            coeff=np.zeros_like(beta) if zero else beta
            with flops.BudgetContext(flop_budget=1e9,quiet=True) as ctx:
                fm=fnp.asarray(mean);fs=fnp.asarray(scale);fb=fnp.asarray(coeff);previous=None;ys=[]
                for l in range(8):
                    previous=fnp_step(fnp,fnp.asarray(raw[l]),fnp.asarray(sigma[l]),fnp.asarray(gate[l]),fnp.asarray(w[l]),previous,fm,fs,fb,l);ys.append(previous)
                out=fnp.stack(ys)
            actual=np.asarray(out).copy();ref=np.zeros_like(expected) if zero else expected
            err=float(np.max(np.abs(actual-ref))/max(float(np.max(np.abs(ref))),1e-12));assert err<=2e-10
            if zero:assert np.array_equal(np.maximum(a['prediction'][-1].astype(np.float64)+actual[-1],0).astype(np.float32),a['prediction'][-1])
            cells.append(dict(zero_beta=zero,relative_error=err,flops=ctx.summary_dict()['flops_used']))
        wide.append(dict(rank=row['rank'],design_action_relative=relative,fnp=cells))
    records.append(dict(rank=row['rank'],record=c,design=pin(path),role='check' if row['rank'] in [35,100] else 'train'))
    print(dict(rank=row['rank'],design_bytes=path.stat().st_size),flush=True)
    del a,raw,sigma,gate,w,b
write(HERE/'designs_sealed.json',dict(passed=True,rows=records,wide_certificates=wide,source_seal=pin(HERE/'fit_sources_sealed.json'),targets_accessed=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(designs=len(records),wide_certificates=wide),flush=True)
