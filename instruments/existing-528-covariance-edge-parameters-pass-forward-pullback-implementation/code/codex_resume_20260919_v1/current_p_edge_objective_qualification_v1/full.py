import gc
import os
import resource
import time
import weakref
from common import *
verify();assert read(HERE/'unit_result.json')['passed'];clean_watch('unit')
assert os.getpriority(os.PRIO_PROCESS,0)>=10
assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
import numpy as np
import torch
import allocator
import model_edge as model
import state_io
assert np.__version__=='1.26.4' and torch.__version__=='2.7.0'
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(9409213)
state_io.initial_rng('reference',torch,np)
meta=read(HERE/'inputs.json');records=[];first35=None;first_grad=None;first_trace=None
recorder=allocator.Recorder(allocator.Allocator(),HERE/'allocation_events.jsonl')
Card=allocator.wrapped_card(model,recorder)
for index,(rank,trainable) in enumerate([(35,False),(35,True),(100,True),(35,True)]):
    print(dict(starting_call=index,rank=rank,edge_gradients=trainable),flush=True);start=time.monotonic()
    wp=meta['weights'][str(rank)];check(wp)
    w=np.load(wp['path'],allow_pickle=False)
    assert w.shape==(16,1024,1024) and w.dtype==np.float32
    p,a,c,e=model.torch_payload(torch,np,trainable)
    def hashes():return dict(p={k:model.tensor_sha(v) for k,v in p.items()},edge={k:model.tensor_sha(v) for k,v in e.items()},a=model.tensor_sha(a),c=model.tensor_sha(c))
    before=hashes();recorder.call=index;recorder.phase='forward'
    card=Card(torch,w,p,a,c,e);ref=weakref.ref(card)
    prediction=card.run(trainable)
    assert prediction.shape==(16,1024) and prediction.dtype==torch.float32
    assert bool(torch.isfinite(prediction).all()) and bool((prediction>=0).all())
    gradients=None
    if trainable:
        recorder.phase='backward'
        direction=torch.cos(torch.arange(1024,dtype=torch.float64)*.013+.4)
        loss=(prediction[-1].to(torch.float64)*direction).mean()*1000
        loss.backward();del direction,loss
        assert all(e[k].grad is not None and bool(torch.isfinite(e[k].grad).all()) for k in EKEYS)
        norm=sum(float(e[k].grad.to(torch.float64).square().sum()) for k in EKEYS)**.5
        assert norm>0
        assert all(v.grad is None for v in p.values())
        assert all(v.grad is None for k,v in e.items() if k not in EKEYS)
        gradients={k:model.tensor_sha(e[k].grad) for k in EKEYS}
    assert hashes()==before
    assert set(card.domain)=={str(l) for l in range(16)}
    assert set(card.states)=={str(l) for l in range(16)}
    assert set(card.selectors)=={f'{l}:{s}' for l in range(15) for s in range(3)}
    trace=dict(domain=card.domain,selectors=card.selectors,states=card.states)
    output=prediction.detach().numpy().copy()
    if index==0:first35=output.tobytes()
    if rank==35:assert output.tobytes()==first35
    if index==1:first_grad=gradients;first_trace=trace
    if index==3:assert gradients==first_grad and trace==first_trace
    if trainable:assert any(v>1 for k,v in card.visits.items() if k.startswith('domain:'))
    path=HERE/f'call_{index:02d}_rank{rank:04d}.npy'
    with path.open('xb') as f:np.save(f,output,allow_pickle=False)
    record=dict(index=index,rank=rank,edge_gradients=trainable,prediction=pin(path),weights=wp,
        trace=trace,visits=card.visits,gradient_hashes=gradients,parameters_unchanged=True,
        parameter_hashes=before,native_derivative_certified=False,targets_decoded=False)
    del prediction,output,card,w
    gc.collect();assert ref() is None
    recorder.phase='retired';record['retirement']=recorder.point('qualification_card_retired')
    assert record['retirement']['after']['rss_bytes']<2_000_000_000
    record['wall_s']=time.monotonic()-start;record['card_retired']=True
    write(HERE/f'call_{index:02d}_result.json',record);records.append(pin(HERE/f'call_{index:02d}_result.json'))
    print(dict(completed_call=index,rank=rank,wall_s=record['wall_s'],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),flush=True)

# Exercise the generic real-leaf/Adam boundary, with no truth-derived loss.
opt=torch.optim.Adam([e[k] for k in EKEYS],lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
opt.step()
boundary=state_io.save(HERE,1,torch,np,e,opt,EKEYS)
p2,a2,c2,e2=model.torch_payload(torch,np,True)
opt2=torch.optim.Adam([e2[k] for k in EKEYS],lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
restored=state_io.restore(Path(boundary['path']),1,torch,np,e2,opt2,EKEYS)
assert state_io.descriptors(state_io.arrays(torch,np,e,opt,EKEYS))==state_io.descriptors(state_io.arrays(torch,np,e2,opt2,EKEYS))
for k in EKEYS:
    gradient=torch.sin(torch.arange(e[k].numel(),dtype=torch.float32)*.07).reshape(e[k].shape)*.001
    e[k].grad=gradient.clone();e2[k].grad=gradient.clone()
opt.step();opt2.step()
assert state_io.descriptors(state_io.arrays(torch,np,e,opt,EKEYS))==state_io.descriptors(state_io.arrays(torch,np,e2,opt2,EKEYS))
assert state_io.groups(opt)==state_io.groups(opt2)
opt.zero_grad(set_to_none=True);opt2.zero_grad(set_to_none=True);opt.state.clear();opt2.state.clear()
del opt,opt2,p,a,c,e,p2,a2,c2,e2,gradient
gc.collect();recorder.point('qualification_complete');recorder.stream.close()
write(HERE/'full_result.json',dict(passed=True,records=records,frozen_checkpoint_primal_exact=True,
    repeated35_primal_trace_gradient_exact=True,all_parameters_unchanged_during_graphs=True,
    edge_values=528,parent_p_frozen=True,state_boundary=boundary,restored_boundary=restored,
    real_adam_boundary_exact=True,next_synthetic_update_exact=True,synthetic_optimizer_updates=2,
    production_payload_exported=False,targets_decoded=False,native_derivative_certified=False,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,full_calls=4,backwards=3,state_boundary_exact=True),flush=True)
