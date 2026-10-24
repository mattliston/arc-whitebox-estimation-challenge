import math,time,json,resource,hashlib,gc,weakref
from common import HERE,TRAIN,read,write,pin,sha,verify
def main():
    verify('learning_sources_sealed.json');verify('fit_inputs_sealed.json')
    g=read(HERE/'prepare_watch.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    prep=read(HERE/'preparation.json');assert prep['completed'] and prep['train_ranks']==TRAIN and not prep['check_target_access']
    assert sha(prep['data']['path'])==prep['data']['sha256']
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(2026092057);started=time.monotonic()
    with np.load(prep['data']['path'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
    X=torch.tensor(data['X']);y=torch.tensor(data['y']);assert X.shape==(49152,102) and y.shape==(49152,)
    p={'W1':torch.nn.Parameter(torch.empty(102,32)),'b1':torch.nn.Parameter(torch.empty(32)),
       'W2':torch.nn.Parameter(torch.zeros(32)),'b2':torch.nn.Parameter(torch.zeros(()))}
    torch.nn.init.kaiming_uniform_(p['W1'].T,a=math.sqrt(5));torch.nn.init.uniform_(p['b1'],-1/math.sqrt(102),1/math.sqrt(102))
    assert sum(v.numel() for v in p.values())==3329
    def forward(x,params):return torch.relu(x@params['W1']+params['b1'])@params['W2']+params['b2']
    with torch.no_grad():assert not forward(X[:17],p).any()
    gen=torch.Generator().manual_seed(2026092057)
    schedule=torch.stack([torch.randperm(len(y),generator=gen) for _ in range(100)])
    schedule_path=HERE/'fit_schedule.npy';np.save(schedule_path,schedule.numpy(),allow_pickle=False)
    schedule_pin=pin(schedule_path)
    opt=torch.optim.AdamW(list(p.values()),lr=.003,betas=(.9,.999),eps=1e-8,weight_decay=1e-4,foreach=False)
    updates=0;log=[];max_grad=0.0
    for ep in range(1,101):
        lr=.0003+(.003-.0003)*.5*(1+math.cos(math.pi*(ep-1)/99))
        for group in opt.param_groups:group['lr']=lr
        total=0.0
        for start in range(0,len(y),2048):
            ix=schedule[ep-1,start:start+2048];opt.zero_grad(set_to_none=True)
            pred=forward(X[ix],p);loss=torch.mean((pred-y[ix])**2);assert torch.isfinite(loss)
            loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in p.values())
            gn=torch.nn.utils.clip_grad_norm_(list(p.values()),5,error_if_nonfinite=True);max_grad=max(max_grad,float(gn))
            opt.step();updates+=1;assert all(torch.isfinite(v).all() for v in p.values())
            total+=float(loss.detach())*len(ix)
            del pred,loss,ix,gn
        assert updates==ep*24 and all(int(opt.state[v]['step'])==updates for v in p.values())
        item=dict(epoch=ep,updates=updates,online_mse=total/len(y),lr=lr);log.append(item)
        if ep==1 or ep%10==0:print(json.dumps(item),flush=True)
    assert updates==2400
    arrays={k:v.detach().numpy().copy() for k,v in p.items()};arrays.update({k:data[k] for k in ['mu','sd','label_rms']})
    terminal_path=HERE/'terminal.npz';np.savez(terminal_path,**arrays);terminal_pin=pin(terminal_path)
    refs=[weakref.ref(v) for v in p.values()];optimizer_ref=weakref.ref(opt)
    opt.zero_grad(set_to_none=True);opt.state.clear()
    for group in opt.param_groups:group['params'].clear()
    opt.param_groups.clear();del group,opt,p;gc.collect()
    assert optimizer_ref() is None and all(ref() is None for ref in refs)
    # Reload the one sealed terminal payload into fresh nontrainable tensors.
    with np.load(terminal_path,allow_pickle=False) as z:frozen={k:torch.tensor(z[k]) for k in ['W1','b1','W2','b2']}
    assert all(not v.requires_grad for v in frozen.values())
    chunks=[]
    with torch.inference_mode():
        for start in range(0,len(y),4096):chunks.append(forward(X[start:start+4096],frozen).numpy().copy())
    prediction=np.concatenate(chunks);np.save(HERE/'train_source_predictions.npy',prediction,allow_pickle=False)
    target=data['y'].astype(np.float64);mse=float(np.mean((prediction.astype(np.float64)-target)**2));zero=float(np.mean(target*target));ratio=mse/zero
    assert sha(terminal_path)==terminal_pin['sha256'] and sha(schedule_path)==schedule_pin['sha256']
    result=dict(completed=True,passed=ratio<=.50,gate='terminal TRAIN normalized local-label MSE / zero <=.50',train_ranks=TRAIN,
        trainable_scalars=3329,epochs=100,updates=updates,device='cpu',torch_version=torch.__version__,numpy_version=np.__version__,
        train_mse=mse,zero_mse=zero,train_ratio=ratio,terminal=terminal_pin,schedule=schedule_pin,
        source_predictions=pin(HERE/'train_source_predictions.npy'),optimizer_and_leaves_retired=True,reloaded_nontrainable=True,
        check_targets_accessed=False,check_forecasts_started=False,max_unclipped_grad_norm=max_grad,epoch_log=log,
        wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'fit_result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='epoch_log'}),flush=True)
if __name__=='__main__':main()
