import importlib.util,resource,sys,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import *
from reference import covariance
seal_check();done('legacy_replay');assert read(HERE/'legacy_replay.json')['passed']
build=read(HERE/'build.json');verify(build['candidate']);verify(build['parameters'])
with np.load(build['parameters']['path'],allow_pickle=False) as z:q={k:z[k].copy() for k in z.files}
with np.load(P0,allow_pickle=False) as z:p={k:z[k].copy() for k in z.files if k!='first'}
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
feat=module('history_original_features',A/'radial_features.py');gnn=module('history_original_gnn',A/'radial_gnn_numpy.py')
rng=np.random.default_rng(870035);rows=[];model=None
for index,n in enumerate((7,9,16)):
    h=rng.normal(0,.15,(n,64)).astype(np.float32);s=np.exp(rng.normal(0,.15,n))
    v=rng.normal(size=(n,n));c=v@v.T+np.eye(n);r=c/np.sqrt(np.outer(np.diag(c),np.diag(c)));pre=r*np.outer(s,s)
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        if model is None:model=module('history_candidate_unit',build['candidate']['path'])
        hh=fnp.asarray(h.copy());aux={'sigma':fnp.asarray(s.copy()),'pre_cov':fnp.asarray(pre.copy())}
        out=model._history_covariance(hh,aux)
    cold_bill=int(ctx.summary_dict()['flops_used']);actual=np.asarray(out).copy()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        hh2=fnp.asarray(h.copy());aux2={'sigma':fnp.asarray(s.copy()),'pre_cov':fnp.asarray(pre.copy())}
        repeat=model._history_covariance(hh2,aux2)
    warm_bill=int(ctx.summary_dict()['flops_used']);assert actual.tobytes()==np.asarray(repeat).tobytes()
    if index==0:assert cold_bill>warm_bill
    else:assert cold_bill==warm_bill
    errors=[relative(actual,covariance(h,r,s,q,wide=v)) for v in (False,True)]
    assert max(errors)<=3e-5 and relative(actual,actual.T)<=3e-5
    order=rng.permutation(n)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        perm=model._history_covariance(fnp.asarray(h[order].copy()),
            {'sigma':fnp.asarray(s[order].copy()),'pre_cov':fnp.asarray(pre[np.ix_(order,order)].copy())})
        oldtheta=model._HISTORY_PARAMETERS['Q']['theta'];model._HISTORY_PARAMETERS['Q']['theta']=fnp.zeros_like(oldtheta)
        zero=model._history_covariance(hh2,aux2);model._HISTORY_PARAMETERS['Q']['theta']=oldtheta
    assert not np.any(np.asarray(zero)) and relative(np.asarray(perm),actual[np.ix_(order,order)])<=3e-5
    rows.append(dict(n=n,numpy_relative=errors[0],fp64_relative=errors[1],cold_flops=cold_bill,warm_flops=warm_bill,
        repeat_bytes_exact=True,zero_theta_exact=True,permutation_passed=True))
history=[];n=7;state=rng.normal(0,.05,(n,64)).astype(np.float32)
for layer in (0,1,8,14):
    w=rng.normal(0,.04,(n,n)).astype(np.float32);s=np.exp(rng.normal(0,.1,n));alpha=rng.normal(0,.2,n)
    v=s*s;pdf=np.exp(-alpha*alpha/2)/np.sqrt(2*np.pi)
    # A coherent-enough bounded synthetic feature domain; both references see the same values.
    gate=np.full(n,.55);mg=s*(pdf+alpha*gate);m=mg+1e-4*rng.normal(size=n)
    zz=rng.normal(size=(n,n));pre=zz@zz.T*.02+np.diag(v)
    C=pre*.4;K=rng.normal(0,.005,(n,n))
    aux=dict(sigma=s,alpha=alpha,pdf=pdf,gate=gate,gaussian_mean=mg,pre_cov=pre,
        pre_k3=rng.normal(0,.003,n),pre_k4=rng.normal(0,.002,n),post_v=np.diag(C).copy(),
        post_k3=rng.normal(0,.003,n),post_k4=rng.normal(0,.002,n))
    expected_x=feat.features(layer,m,C,K,aux,w)
    with flops.BudgetContext(flop_budget=2**41,quiet=True):
        ax={k:fnp.asarray(a.copy()) for k,a in aux.items()}
        x=model.features(layer,fnp.asarray(m.copy()),fnp.asarray(C.copy()),fnp.asarray(K.copy()),ax,fnp.asarray(w.copy()))
        delta,hnext=model.gnn_step(layer,x,ax['sigma'],fnp.asarray(w.copy()),fnp.asarray(state.copy()),model._history_parameters()['P'])
    xn=np.asarray(x).copy();hn=np.asarray(hnext).copy();_,href=gnn.gnn_step(layer,expected_x,s,w,state,p)
    er=float(np.max(np.abs(xn-expected_x))/max(1,float(np.max(np.abs(expected_x)))))
    hr=relative(hn,href);assert er<=3e-6 and hr<=3e-5,(layer,er,hr)
    history.append(dict(layer=layer,feature_scaled_error=er,history_relative=hr));state=hn
assert model._HISTORY_PARAMETERS is not None
write(HERE/'unit.json',dict(passed=True,kernel=rows,history=history,model=build['model'],P0=build['P0'],
    production_parameters_restored=True,ordinary_1B_targets_read=False,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,kernel=rows,history=history),flush=True)
