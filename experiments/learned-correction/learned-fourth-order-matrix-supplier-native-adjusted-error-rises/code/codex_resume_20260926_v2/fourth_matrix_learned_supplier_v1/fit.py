from common import *
import math
import time
import resource
import numpy as np
import torch
from model import Supplier
torch.set_num_threads(1);torch.set_num_interop_threads(1)
verify();r=read(HERE/'prepared.json');g=read(HERE/'prepare_data_guard.json')
assert r['completed'] and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
check(r['dataset']);check(r['normalizers'])
with np.load(r['dataset']['path'],allow_pickle=False) as z:d={k:torch.from_numpy(z[k]) for k in z.files}
torch.manual_seed(20260927392);model=Supplier();rng=np.random.default_rng(20260927392)
optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4,foreach=False)
N=len(d['target']);epochs=60;batch=4096;steps=epochs*math.ceil(N/batch);step=0;history=[];started=time.monotonic()
with (HERE/'training.jsonl').open('x') as stream:
    for epoch in range(epochs):
        order=rng.permutation(N);pairloss=diagloss=0.;count=0
        for lo in range(0,N,batch):
            indices=torch.from_numpy(order[lo:lo+batch]);node=torch.from_numpy(rng.integers(len(d['F']),size=1024,dtype=np.int64))
            phase=step/max(steps-1,1);lr=.00005+(.002-.00005)*.5*(1+math.cos(math.pi*phase))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            pred=model.pair(d['F'][d['i'][indices]],d['F'][d['j'][indices]],d['edges'][indices])
            lp=torch.mean((pred-d['target'][indices])**2);ld=torch.mean((model.diag(d['F'][node])-d['diagonal'][node])**2)
            loss=.5*(lp+ld);assert torch.isfinite(loss);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(grad)
            optimizer.step();pairloss+=float(lp.detach())*len(indices);diagloss+=float(ld.detach())*len(indices);count+=len(indices);step+=1
        row=dict(epoch=epoch+1,updates=step,pair_training_loss=pairloss/count,diagonal_training_loss=diagloss/count,
                 learning_rate=lr,elapsed_s=time.monotonic()-started)
        history.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
        print(json.dumps(row),flush=True)
assert step==steps
payload={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
with (HERE/'model.npz').open('xb') as f:np.savez(f,**payload)
record=dict(completed=True,terminal_epoch=60,updates=step,history=history,model=pin(HERE/'model.npz'),normalizers=r['normalizers'],
            dataset=r['dataset'],validation_labels_accessed=False,elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seal_sha256=sha(HERE/'seal.json'))
once(HERE/'training_result.json',record)
once(HERE/'model_frozen.json',dict(model=record['model'],normalizers=record['normalizers'],training=pin(HERE/'training_result.json'),
                               terminal_epoch=60,validation_labels_accessed=False,seal_sha256=sha(HERE/'seal.json')))
print(json.dumps(dict(event='model_frozen',model_sha256=record['model']['sha256'],elapsed_s=record['elapsed_s'])),flush=True)
