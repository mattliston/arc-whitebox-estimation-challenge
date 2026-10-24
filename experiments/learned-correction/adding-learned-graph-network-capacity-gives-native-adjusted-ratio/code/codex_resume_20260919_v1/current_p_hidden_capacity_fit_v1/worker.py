import argparse
import gc
import math
import os
import resource
import time
import weakref
from fit_common import *
import state_io

verify()
parser=argparse.ArgumentParser();parser.add_argument('update',type=int);parser.add_argument('--resume',type=Path)
args=parser.parse_args();assert 0<=args.update<40
BASE=HERE;folder=BASE/f'batch_{args.update:02d}';folder.mkdir(exist_ok=False)
if args.update:assert args.resume is not None and args.resume.resolve()==BASE/f'batch_{args.update-1:02d}'/f'checkpoint_{args.update:02d}.json'
else:assert args.resume is None
assert os.getpriority(os.PRIO_PROCESS,0)>=10
assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
import numpy as np
import torch
assert np.__version__=='1.26.4' and torch.__version__=='2.7.0'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
model,allocation=qualified_modules();inputs=read(BASE/'inputs.json')
order=np.random.Generator(np.random.PCG64DXSM(9409215)).permutation(inputs['train']).tolist()
assert len(order)==len(set(order))==160 and set(order)==set(inputs['train'])
write(folder/'training_order.json',dict(seed=9409215,ranks=order,updates=40,batch_networks=4,targets_decoded=False))
p,a,c,edge=model.torch_payload(torch,np,True)
fixed_p={k:model.tensor_sha(v) for k,v in p.items()}
fixed_ac=[model.tensor_sha(a),model.tensor_sha(c)]
fixed_edge={k:model.tensor_sha(v) for k,v in edge.items() if k not in EKEYS}
initial_edge={k:model.tensor_sha(v) for k,v in edge.items()}
parameters=[edge[k] for k in EKEYS]
assert sum(v.numel() for v in parameters)==12352 and all(v.requires_grad for v in parameters)
opt=torch.optim.Adam(parameters,lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
state_io.initial_rng('split0' if args.update==0 else 'continuation',torch,np)
restored=None
if args.resume is not None:restored=state_io.restore(args.resume,args.update,torch,np,edge,opt,EKEYS)
def frozen_check():
    assert {k:model.tensor_sha(v) for k,v in p.items()}==fixed_p
    assert all(not v.requires_grad and v.grad is None for v in p.values())
    assert [model.tensor_sha(a),model.tensor_sha(c)]==fixed_ac
    assert {k:model.tensor_sha(v) for k,v in edge.items() if k not in EKEYS}==fixed_edge
    assert all(not v.requires_grad and v.grad is None for k,v in edge.items() if k not in EKEYS)
    assert all(bool(torch.isfinite(v).all()) for v in edge.values())
frozen_check()
recorder=allocation.Recorder(allocation.Allocator(),folder/'allocation_events.jsonl')
Card=allocation.wrapped_card(model,recorder);started=time.monotonic();records=[]
lr=1e-4+.5*(1e-3-1e-4)*(1+math.cos(math.pi*args.update/39))
opt.param_groups[0]['lr']=lr;opt.zero_grad(set_to_none=True)
batch=order[args.update*4:args.update*4+4]
for within,rank in enumerate(batch):
    index=args.update*4+within;start=time.monotonic();resource_floor()
    print(dict(starting_graph=index,rank=rank,update=args.update+1),flush=True)
    assert rank in inputs['train'] and rank not in inputs['valid']
    wp,tp=inputs['weights'][str(rank)],inputs['targets'][str(rank)];check(wp);check(tp)
    w=np.load(wp['path'],allow_pickle=False)
    with np.load(tp['path'],allow_pickle=False) as z:truth_np=z['mc_post'].copy()
    assert w.shape==(16,1024,1024) and w.dtype==np.float32
    assert truth_np.shape==(16,1024) and np.isfinite(truth_np).all()
    truth=torch.tensor(truth_np,dtype=torch.float64)
    before={k:model.tensor_sha(v) for k,v in edge.items()}
    recorder.call=index;recorder.phase='forward';recorder.point('training_call_start')
    card=Card(torch,w,p,a,c,edge);ref=weakref.ref(card);prediction=card.run(True)
    error=prediction.to(torch.float64)-truth
    final=error[-1].square().mean();earlier=error[:-1].square().mean()
    loss=(final+.05*earlier)/2e-8/4;assert bool(torch.isfinite(loss))
    recorder.phase='backward';loss.backward();recorder.point('backward_complete')
    assert all(edge[k].grad is not None and bool(torch.isfinite(edge[k].grad).all()) for k in EKEYS)
    frozen_check();assert {k:model.tensor_sha(v) for k,v in edge.items()}==before
    assert set(card.domain)=={str(l) for l in range(16)}
    assert set(card.states)=={str(l) for l in range(16)}
    assert set(card.selectors)=={f'{l}:{s}' for l in range(15) for s in range(3)}
    assert any(v>1 for k,v in card.visits.items() if k.startswith('domain:'))
    output=prediction.detach().numpy().copy()
    assert output.shape==(16,1024) and output.dtype==np.float32 and np.isfinite(output).all() and (output>=0).all()
    path=folder/f'train_{index:03d}_rank{rank:04d}.npy'
    with path.open('xb') as f:np.save(f,output,allow_pickle=False)
    row=dict(index=index,rank=rank,update=args.update+1,within_batch=within,prediction=pin(path),weights=wp,target=tp,
        final_mse=float(final.detach()),earlier_mse=float(earlier.detach()),batch_scaled_loss=float(loss.detach()),
        edge_hashes_during_graph=before,accumulated_gradient_hashes={k:model.tensor_sha(edge[k].grad) for k in EKEYS},
        domain=card.domain,selectors=card.selectors,states=card.states,visits=card.visits,
        fixed_parameters_unchanged=True,validation_targets_accessed=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    del prediction,error,final,earlier,loss,truth,truth_np,output,card,w
    gc.collect();assert ref() is None
    recorder.phase='retired';row['retirement_memory']=recorder.point('training_card_retired')
    assert row['retirement_memory']['after']['rss_bytes']<2_000_000_000
    row['card_retired']=True;row['wall_s']=time.monotonic()-start
    path=folder/f'train_{index:03d}_result.json';write(path,row);records.append(pin(path))
    print(dict(completed_graph=index,rank=rank,final_mse=row['final_mse'],wall_s=row['wall_s'],peak_rss_bytes=row['peak_rss_bytes']),flush=True)
norm=float(torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True,foreach=False))
clipped={k:model.tensor_sha(edge[k].grad) for k in EKEYS};opt.step();frozen_check()
assert len(opt.state)==3 and all(all(bool(torch.isfinite(v).all()) for v in s.values()) for s in opt.state.values())
boundary=state_io.save(folder,args.update+1,torch,np,edge,opt,EKEYS)
update=dict(update=args.update+1,ranks=batch,learning_rate=lr,pre_clip_gradient_norm=norm,
    clipped_gradient_hashes=clipped,edge_hashes={k:model.tensor_sha(v) for k,v in edge.items()},
    fixed_parameters_unchanged=True,checkpoint=boundary)
write(folder/f'update_{args.update+1:02d}.json',update)
opt.zero_grad(set_to_none=True);opt.state.clear();parameters.clear();gc.collect()
recorder.point('worker_complete');recorder.stream.close();verify()
write(folder/'result.json',dict(completed=True,passed=True,update=args.update+1,training_graphs=4,records=records,
    update_receipt=pin(folder/f'update_{args.update+1:02d}.json'),checkpoint=boundary,restored_checkpoint=restored,
    initial_edge_hashes=initial_edge,terminal_edge_hashes=update['edge_hashes'],fixed_p_hashes=fixed_p,
    fixed_ac_hashes=fixed_ac,fixed_edge_hashes=fixed_edge,hidden_trainable_keys=list(EKEYS),hidden_trainable_values=12352,
    fixed_parameters_unchanged=True,validation_targets_accessed=False,checkpoint_selection=False,
    source_seal=pin(BASE/'sources_sealed.json'),native_derivative_certified=False,wall_s=time.monotonic()-started))
print(dict(completed_update=args.update+1,gradient_norm=norm,learning_rate=lr,wall_s=time.monotonic()-started),flush=True)
