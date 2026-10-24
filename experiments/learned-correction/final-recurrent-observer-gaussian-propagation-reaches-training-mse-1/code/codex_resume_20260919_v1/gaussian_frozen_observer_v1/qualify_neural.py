"""No-target CPU/MPS, native and gradient checks for the fixed architecture."""
import time,resource,gc
from common import HERE,read,write,verify,pin,enc4_direction
def main():
    import numpy as np
    import torch
    import model
    rt=model.runtime();verify('neural_sources_sealed.json')
    g=read(HERE/'fnp_replay_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    refs=read(HERE/'fnp_replay/result.json');assert refs['passed']
    inp=read(HERE/'inputs.json');row=next(r for r in inp['rows'] if r['rank']==35)
    with np.load(inp['initial_P']['path'],allow_pickle=False) as z:P={k:z[k].copy() for k in z.files}
    with np.load(row['features']['path'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    W=np.load(row['weights']['path'],allow_pickle=False);out=HERE/'neural_qualification';out.mkdir(exist_ok=False)
    start=time.monotonic();records=[];driver=[]
    def diagnostic(sign):
        q={k:v.copy() for k,v in P.items()};q['head'][15]=(.02*np.sin(.13*np.arange(1,65,dtype=np.float64))).astype(np.float32)
        q['head_b'][15]=.01;q['gain'][15]=.03;q['enc4_w']=(q['enc4_w']+np.float32(sign*2**-8)*enc4_direction(np)).astype(np.float32)
        return q
    for device in ('cpu','mps'):
        data=model.inputs(arrays,W,device)
        for name,sign in [('nonzero',0),('plus',1),('minus',-1)]:
            q=diagnostic(sign);params,fixed=model.initialize(q,device,trainable=False)
            with torch.no_grad():served,delta,states=model.forward(data,params,fixed,True)
            if device=='mps':torch.mps.synchronize();driver.append(int(torch.mps.driver_allocated_memory()))
            reference=next(r for r in refs['records'] if r['name']==name)
            with np.load(reference['output']['path'],allow_pickle=False) as z:ys=z['served'];yd=z['delta']
            actual=served.cpu().numpy();err=actual.astype(np.float64)-ys.astype(np.float64)
            finite=bool(torch.isfinite(delta).all()) and all(bool(torch.isfinite(v).all()) for v in states)
            rec=dict(device=device,name=name,max_abs=float(np.max(np.abs(err))),rms=float(np.sqrt(np.mean(err*err))),delta_max_abs=float(np.max(np.abs(delta.cpu().numpy().astype(np.float64)-yd))),finite=finite)
            rec['passed']=finite and rec['max_abs']<=1e-6 and rec['rms']<=1e-7;records.append(rec)
            assert rec['passed'],rec
            del params,fixed,served,delta,states
        del data;model.retire()
    rng=np.random.default_rng(2026092053);n=7
    toy=dict(features=rng.normal(size=(16,n,30)).astype(np.float64),sigma=np.ones((16,n),np.float64),final_host=np.zeros(n,np.float64))
    weights=(rng.normal(size=(16,n,n))/np.sqrt(n)).astype(np.float64);Q=diagnostic(0)
    Q['centers']=np.zeros_like(Q['centers']);Q['scales']=np.ones_like(Q['scales'])
    data=model.inputs(toy,weights,'cpu',torch.float64);params,fixed=model.initialize(Q,'cpu',dtype=torch.float64)
    vector=torch.tensor(np.sin(.23*np.arange(1,n+1)),dtype=torch.float64)
    _,delta,_=model.forward(data,params,fixed);loss=(delta*vector).sum();loss.backward();tiny=[]
    for index,(key,value) in enumerate(params.items()):
        direction=torch.sin(torch.arange(1,value.numel()+1,dtype=torch.float64)*(.11+.013*index)).reshape(value.shape)/np.sqrt(value.numel())
        grad=float((value.grad*direction).sum());eps=1e-5;values=[]
        for sign in (1,-1):
            altered={k:v.detach().clone() for k,v in params.items()};altered[key]+=sign*eps*direction
            with torch.no_grad():_,d,_=model.forward(data,altered,fixed);values.append(float((d*vector).sum()))
        fd=(values[0]-values[1])/(2*eps);relative=abs(fd-grad)/max(abs(fd),abs(grad),1e-15)
        rec=dict(parameter=key,gradient=grad,finite_difference=fd,relative=relative,passed=relative<=1e-5 and abs(grad)>1e-15);tiny.append(rec);assert rec['passed'],rec
    del data,params,fixed,loss,delta;gc.collect()
    gradients={}
    for device in ('cpu','mps'):
        data=model.inputs(arrays,W,device);params,fixed=model.initialize(diagnostic(0),device)
        vector=torch.tensor(np.sin(.13*np.arange(1,1025))/np.sqrt(1024),dtype=torch.float32,device=device)
        _,delta,_=model.forward(data,params,fixed);(delta*vector).sum().backward()
        if device=='mps':torch.mps.synchronize();driver.append(int(torch.mps.driver_allocated_memory()))
        gradients[device]=np.concatenate([v.grad.detach().cpu().numpy().ravel() for v in params.values()]).astype(np.float64)
        assert np.isfinite(gradients[device]).all() and np.linalg.norm(gradients[device])>0
        del data,params,fixed,delta,vector;model.retire()
    relative=float(np.linalg.norm(gradients['mps']-gradients['cpu'])/np.linalg.norm(gradients['cpu']));assert relative<=.01,relative
    assert max(driver)<=8_000_000_000
    result=dict(passed=True,records=records,tiny_gradient=tiny,gradient_cpu_mps_relative=relative,runtime=rt,
                sampled_mps_driver_peak_bytes=max(driver),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                wall_s=time.monotonic()-start,truth_accessed=False)
    write(out/'result.json',result);print({'passed':True,'max_native_error':max(r['max_abs'] for r in records),'gradient_relative':relative,'wall_s':result['wall_s']},flush=True)
if __name__=='__main__':main()
