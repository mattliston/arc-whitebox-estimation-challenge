"""Frozen-bank parity, complete recurrent gradients, and real optimizer transport."""
import ast
import gc
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
import model_basis as model
import model_bank as bank_model
import state_io
assert np.__version__=='1.26.4' and torch.__version__=='2.7.0'
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(9409221)
state_io.initial_rng('reference',torch,np)
bank_model.SOURCE=BANK;bank_model.PARENT_SHA=BANK_SHA
meta=read(HERE/'inputs.json');records=[]
baseline_output=None;baseline_trace=None;baseline_parameters=None
first_grad=None;first_trace=None;sentinel_output=None;sentinel_grad=None;sentinel_trace=None
import base64,zlib
unit=read(HERE/'unit_result.json');assert unit['passed']
packed=[ast.literal_eval(n.value) for n in ast.parse((BI/'sentinel.py').read_text()).body
    if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_JR_LITERAL']
assert len(packed)==1
sentinel={'jr_'+k:np.frombuffer(zlib.decompress(base64.b64decode(blob,validate=True)),dtype=np.float32).reshape(shape).copy() for k,(shape,blob) in packed[0].items()}
assert set(sentinel)==set(EKEYS) and all(np.isfinite(v).all() for v in sentinel.values())
recorder=allocator.Recorder(allocator.Allocator(),HERE/'allocation_events.jsonl')
Cards={'bank':allocator.wrapped_card(bank_model,recorder),'readout':allocator.wrapped_card(model,recorder)}
for index,(rank,trainable,mode) in enumerate(meta['full_call_order']):
    print(dict(starting_call=index,rank=rank,readout_gradients=trainable,mode=mode),flush=True);start=time.monotonic()
    wp=meta['weights'][str(rank)];check(wp);w=np.load(wp['path'],allow_pickle=False)
    assert w.shape==(16,1024,1024) and w.dtype==np.float32
    engine=bank_model if mode=='bank' else model
    assert mode in ('bank','zero','sentinel')
    p,a,c,e=engine.torch_payload(torch,np,trainable)
    assert all(not value.requires_grad for value in p.values())
    assert not a.requires_grad and not c.requires_grad
    assert all(not value.requires_grad for key,value in e.items() if key not in EKEYS)
    if mode=='bank':assert not trainable and not any(k.startswith('jr_') for k in e)
    else:
        assert set(k for k,v in e.items() if v.requires_grad)==(set(EKEYS) if trainable else set())
        assert sum(e[k].numel() for k in EKEYS)==3169
        for k in ('jr_theta','jr_phi','jr_head','jr_head_b'):assert bool((e[k]==0).all())
        assert e['jr_basis'].shape==(64,48) and torch.equal(e['jr_basis'].detach(),torch.eye(64,48))
    if mode=='sentinel':
        with torch.no_grad():
            for k in EKEYS:e[k].copy_(torch.from_numpy(sentinel[k]))
    def hashes():return dict(p={k:model.tensor_sha(v) for k,v in p.items()},edge={k:model.tensor_sha(v) for k,v in e.items()},a=model.tensor_sha(a),c=model.tensor_sha(c))
    before=hashes();recorder.call=index;recorder.phase='forward'
    card=Cards['bank' if mode=='bank' else 'readout'](torch,w,p,a,c,e);ref=weakref.ref(card)
    card.gnn={};original_gnn=card.env['gnn_step']
    # Bind the current Card explicitly and release the observer before retirement.
    def observed_gnn(layer,x,sigma,weight,state,parameters,owner=card,original=original_gnn):
        incoming=model.tensor_sha(state);answer=original(layer,x,sigma,weight,state,parameters)
        owner.observe('gnn',layer,dict(incoming_state_sha=incoming,outgoing_state_sha=model.tensor_sha(answer[1]),delta_sha=model.tensor_sha(answer[0])))
        return answer
    card.env['gnn_step']=observed_gnn
    prediction=card.run(trainable)
    assert prediction.shape==(16,1024) and prediction.dtype==torch.float32
    assert bool(torch.isfinite(prediction).all()) and bool((prediction>=0).all())
    gradients=None;gradient_norm=None
    if trainable:
        recorder.phase='backward'
        direction=torch.cos(torch.arange(1024,dtype=torch.float64)*.013+.4)
        loss=(prediction[-1].to(torch.float64)*direction).mean()*1000
        loss.backward();del direction,loss
        assert all(e[k].grad is not None and bool(torch.isfinite(e[k].grad).all()) for k in EKEYS)
        gradient_norm=sum(float(e[k].grad.to(torch.float64).square().sum()) for k in EKEYS)**.5
        assert gradient_norm>0
        norms={k:float(e[k].grad.to(torch.float64).square().sum())**.5 for k in EKEYS}
        if mode=='zero':
            assert norms['jr_basis']==0
            assert all(norms[k]>0 for k in EKEYS if k!='jr_basis')
        else:assert all(v>0 for v in norms.values())

        gradients={k:model.tensor_sha(e[k].grad) for k in EKEYS}
        assert any(v>1 for k,v in card.visits.items() if k.startswith('domain:'))
    assert all(v.grad is None for v in p.values()) and a.grad is None and c.grad is None
    assert all(v.grad is None for k,v in e.items() if k not in EKEYS)
    assert hashes()==before
    assert set(card.domain)==set(card.states)==set(card.gnn)=={str(l) for l in range(16)}
    assert set(card.selectors)=={f'{l}:{s}' for l in range(15) for s in range(3)}
    trace=dict(domain=card.domain,selectors=card.selectors,states=card.states,gnn=card.gnn)
    joint_trace=None
    if mode!='bank':
        assert set(card.jr)=={str(l) for l in range(8,15)}
        for layer,observed in card.jr.items():
            assert observed['incoming_state_sha']==card.gnn[layer]['incoming_state_sha']
            assert observed['cohorts']==int(layer)-7
        joint_trace=dict(card.jr)

    output=prediction.detach().numpy().copy()
    if mode=='bank':
        assert index==0;baseline_output=output.tobytes();baseline_trace=trace;baseline_parameters=before
    else:
        assert before['p']==baseline_parameters['p']
        assert {k:v for k,v in before['edge'].items() if not k.startswith('jr_')}==baseline_parameters['edge']
        assert before['a']==baseline_parameters['a'] and before['c']==baseline_parameters['c']
        if rank==35 and mode=='zero':assert output.tobytes()==baseline_output and trace==baseline_trace
    if mode=='sentinel':
        assert output[:8].tobytes()==baseline_output[:8*1024*4] and output.tobytes()!=baseline_output
        assert all(trace['gnn'][str(l)]==baseline_trace['gnn'][str(l)] for l in range(8))
        if sentinel_output is None:sentinel_output=output.tobytes();sentinel_grad=gradients;sentinel_trace=trace
        else:assert output.tobytes()==sentinel_output and gradients==sentinel_grad and trace==sentinel_trace
    if index==2:first_grad=gradients;first_trace=trace
    if index==4:assert gradients==first_grad and trace==first_trace
    path=HERE/f'call_{index:02d}_rank{rank:04d}.npy'
    with path.open('xb') as f:np.save(f,output,allow_pickle=False)
    record=dict(index=index,rank=rank,readout_gradients=trainable,mode=mode,prediction=pin(path),weights=wp,
        trace=trace,joint_trace=joint_trace,visits=card.visits,gradient_hashes=gradients,gradient_norm=gradient_norm,
        per_leaf_gradient_norms=norms if trainable else None,
        parameters_unchanged=True,parameter_hashes=before,native_derivative_certified=False,targets_decoded=False)
    card.env['gnn_step']=original_gnn
    del observed_gnn,original_gnn,prediction,output,card,w
    gc.collect();assert ref() is None
    recorder.phase='retired';record['retirement']=recorder.point('qualification_card_retired')
    assert record['retirement']['after']['rss_bytes']<2_000_000_000
    record['wall_s']=time.monotonic()-start;record['card_retired']=True
    write(HERE/f'call_{index:02d}_result.json',record);records.append(pin(HERE/f'call_{index:02d}_result.json'))
    print(dict(completed_call=index,rank=rank,wall_s=record['wall_s'],peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),flush=True)

# These synthetic states cannot initialize a fit or candidate.
opt=torch.optim.Adam([e[k] for k in EKEYS],lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False)
opt.step();boundary=state_io.save(HERE,1,torch,np,e,opt,EKEYS)
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
write(HERE/'full_result.json',dict(passed=True,records=records,full_calls=7,synthetic_backwards=5,
    frozen_bank_zero_checkpoint_primal_and_all_traces_exact=True,repeated35_primal_trace_gradient_exact=True,
    all_parameters_unchanged_during_graphs=True,readout_values=3169,hidden_feature_parameters_frozen=True,basis_learnable=True,null_basis_gradient_zero=True,nonzero_basis_gradient_nonzero=True,
    parent_p_A_cstar_edge_and_trained_context_frozen=True,all_trainable_gradients_finite_nonzero=True,
    nonzero_mode_repeat_output_gradient_trace_exact=True,nonzero_mode_first8_layers_exact=True,
    joint_input_state_matches_same_layer_gnn_incoming=True,
    state_boundary=boundary,restored_boundary=restored,real_adam_boundary_exact=True,next_synthetic_update_exact=True,
    synthetic_optimizer_updates=2,production_payload_exported=False,targets_decoded=False,native_derivative_certified=False,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,full_calls=7,backwards=5,state_boundary_exact=True),flush=True)
