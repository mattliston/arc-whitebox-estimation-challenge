import json,time,resource,gc
from common import HERE,verify,write,read,sha
from model import inputs,forward
def main():
    verify('learning_sources_sealed.json');q=read(HERE/'unit_fnp.json');assert q['passed'] and sha(q['fixture']['path'])==q['fixture']['sha256']
    g=read(HERE/'unit_fnp_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);started=time.monotonic()
    with np.load(q['fixture']['path'],allow_pickle=False) as z:f={k:z[k] for k in z.files}
    x=inputs(f['features'],f['hidden'],11,f['mu'],f['sd'],np)
    p={k:f[k] for k in ('W1','b1','W2','b2')}
    pt={k:torch.tensor(v,dtype=torch.float32) for k,v in p.items()};xt=torch.tensor(x)
    actual=(torch.relu(xt@pt['W1']+pt['b1'])@pt['W2']+pt['b2']).numpy();expected=forward(x,p,np)
    output_error=float(np.max(np.abs(actual-expected))/(1+np.max(np.abs(expected))));assert output_error<=2e-5
    pd={k:torch.tensor(v,dtype=torch.float64,requires_grad=True) for k,v in p.items()};xd=torch.tensor(x,dtype=torch.float64)
    target=torch.linspace(-.2,.3,len(x),dtype=torch.float64)
    pred=torch.relu(xd@pd['W1']+pd['b1'])@pd['W2']+pd['b2'];loss=((pred-target)**2).mean();loss.backward()
    pn={k:v.astype(np.float64) for k,v in p.items()};tn=target.numpy();checks=[];eps=1e-6
    choices=[('W1',(0,0)),('W1',(101,2)),('W1',(55,14)),('W1',(7,1)),('b1',(0,)),('b1',(1,)),('W2',(0,)),('W2',(1,)),('b2',())]
    for key,ix in choices:
        plus={k:v.copy() for k,v in pn.items()};minus={k:v.copy() for k,v in pn.items()};plus[key][ix]+=eps;minus[key][ix]-=eps
        finite=(np.mean((forward(x.astype(np.float64),plus,np)-tn)**2)-np.mean((forward(x.astype(np.float64),minus,np)-tn)**2))/(2*eps)
        analytic=float(pd[key].grad[ix]);err=abs(finite-analytic)/max(abs(finite),abs(analytic),1e-8)
        assert err<=2e-4;checks.append(dict(key=key,index=list(ix),relative=err,analytic=analytic,finite=finite))
    assert sum(v.numel() for v in pd.values())==3329 and all(torch.isfinite(v.grad).all() for v in pd.values())
    result=dict(passed=True,torch_version=torch.__version__,numpy_version=np.__version__,device='cpu',output_scaled_error=output_error,
        parameter_gradient_checks=checks,parameters=3329,truth_accessed=False,wall_s=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'unit_torch.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
