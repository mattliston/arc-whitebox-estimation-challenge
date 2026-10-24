import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import HERE,TRAIN,read,write,pin,verify,seal_check
from observer_math import raw_features,raw_design,normalize_design,direct,normalizers,fnp_step
from observer_math_v2 import normalized_design

seal_check('fit_sources_sealed_v2.json');assert read(HERE/'unit.json')['passed']
rng=np.random.Generator(np.random.PCG64(66771));raw=rng.normal(size=(4,7,5));raw[:,:,2]=1.
sigma=np.exp(.1*rng.normal(size=(4,7)));gate=rng.uniform(.1,.9,size=(4,7));weights=rng.normal(size=(4,7,7))/4
mean,scale=normalizers([raw]);beta=rng.normal(size=9)
stable=normalized_design(raw,sigma,gate,weights,mean,scale,True)
old=normalize_design(raw_design(raw,sigma,gate,weights),mean,scale)
assert np.count_nonzero(stable[:,:,2])==0
expected=direct(raw,sigma,gate,weights,mean,scale,beta)
assert np.allclose(stable@beta,expected,rtol=2e-13,atol=2e-13)
train=read(HERE/'captures_sealed.json')['rows'];data=[]
for row in train:
    verify(row['record']['capture'])
    with np.load(row['record']['capture']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    data.append(raw_features(a))
mu,sd=normalizers(data);del data
checks=[]
for rank,folder in [(35,'capture_triple'),(100,'capture_cold100')]:
    c=read(HERE/folder/'result.json')['records'][0]
    with np.load(c['capture']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    raw=raw_features(a);sigma=a['sigma'][8:];gate=a['features'][8:,:,5].astype(np.float64);w=np.load(c['weights']['path'],mmap_mode='r',allow_pickle=False)[8:]
    b=normalized_design(raw,sigma,gate,w,mu,sd,True)
    assert np.count_nonzero(b[:,:,2])==0 and mu[2]==1. and sd[2]==1e-6
    beta=1e-6*np.sin(.13*np.arange(1,103));ref=direct(raw,sigma,gate,w,mu,sd,beta)
    action=float(np.max(np.abs(b@beta-ref))/max(float(np.max(np.abs(ref))),1e-12));assert action<=2e-10
    cells=[]
    for zero in [False,True]:
        coeff=np.zeros_like(beta) if zero else beta
        with flops.BudgetContext(flop_budget=1e9,quiet=True) as ctx:
            fm=fnp.asarray(mu);fs=fnp.asarray(sd);fb=fnp.asarray(coeff);previous=None;ys=[]
            for l in range(8):
                previous=fnp_step(fnp,fnp.asarray(raw[l]),fnp.asarray(sigma[l]),fnp.asarray(gate[l]),fnp.asarray(w[l]),previous,fm,fs,fb,l);ys.append(previous)
            out=fnp.stack(ys)
        actual=np.asarray(out).copy();expected=np.zeros_like(ref) if zero else ref
        err=float(np.max(np.abs(actual-expected))/max(float(np.max(np.abs(expected))),1e-12));assert err<=2e-10
        if zero:assert np.count_nonzero(actual)==0
        cells.append(dict(zero_beta=zero,relative_error=err,flops=ctx.summary_dict()['flops_used']))
    checks.append(dict(rank=rank,normalized_design_action_relative=action,constant_column_exact_zero=True,fnp=cells))
write(HERE/'unit_v2.json',dict(passed=True,source_seal=pin(HERE/'fit_sources_sealed_v2.json'),
    manufactured_old_constant_residue_max=float(np.max(np.abs(old[:,:,2]))),manufactured_stable_constant_exact_zero=True,
    wide_checks=checks,normalizers_from_train_only=True,target_arrays_accessed=False))
print(dict(passed=True,wide_checks=checks),flush=True)
