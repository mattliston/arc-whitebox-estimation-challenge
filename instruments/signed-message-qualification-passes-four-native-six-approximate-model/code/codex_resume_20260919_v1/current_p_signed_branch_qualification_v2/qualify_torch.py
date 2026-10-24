import gc
import resource
import time
import weakref
from common import *
admission()
assert read(HERE / 'native_qualification.json')['passed']
import numpy as np
import torch
import model_parent as parent
import model_branch as branch
from allocator import Allocator, Recorder, wrapped_card
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
began = time.monotonic()
inputs = read(HERE / 'inputs.json')
recorder = Recorder(Allocator(), HERE / 'allocation_events.jsonl')
old_p, old_a, old_c, old_e = parent.torch_payload(torch, np, False)
p, a, c, edge = branch.torch_payload(torch, np, True)
initial = {k:branch.tensor_sha(v) for k,v in p.items()}
old_initial = {k:parent.tensor_sha(v) for k,v in old_p.items()}
fixed = [branch.tensor_sha(v) for v in [a,c] + list(edge.values())]
old_fixed = [parent.tensor_sha(v) for v in [old_a,old_c] + list(old_e.values())]
assert fixed == old_fixed and all(initial[k] == v for k,v in old_initial.items())
optimizer = torch.optim.Adam([p[k] for k in PKEYS], lr=0., betas=(.9,.999), eps=1e-8, foreach=False)
rng = np.random.default_rng(940921)
cotangents = {rank:torch.tensor(rng.normal(size=(16,1024)).astype(np.float64)/1024) for rank in (35,100)}
records, controls, gradient35 = [], {}, None
sequence = [('parent',35,False),('parent',100,False),('branch',35,False),
            ('branch',35,True),('branch',100,True),('branch',35,True)]
for index,(role,rank,checkpointed) in enumerate(sequence):
    print(dict(starting_call=index, role=role, rank=rank, checkpointed=checkpointed), flush=True)
    start = time.monotonic()
    recorder.call, recorder.phase = index, 'setup'
    recorder.point('call_start')
    wp = inputs['weights'][str(rank)]
    check(wp)
    weights = np.load(wp['path'], allow_pickle=False)
    model = parent if role == 'parent' else branch
    Card = wrapped_card(model,recorder)
    args = (old_p,old_a,old_c,old_e) if role == 'parent' else (p,a,c,edge)
    card = Card(torch,weights,*args)
    ref = weakref.ref(card)
    optimizer.zero_grad(set_to_none=True)
    recorder.phase = 'forward'
    if checkpointed:
        predicted = card.run(True)
        recorder.point('forward_complete')
        recorder.phase = 'backward'
        loss = (predicted.to(torch.float64)*cotangents[rank]).sum()
        loss.backward()
        recorder.point('backward_complete')
        assert all(p[k].grad is not None and bool(torch.isfinite(p[k].grad).all()) for k in PKEYS)
        assert float(p['ungated_w'].grad.norm()) > 0
        for key in INACTIVE:
            assert int(torch.count_nonzero(p[key].grad[0])) == 0
        gradients = {k:branch.tensor_sha(p[k].grad) for k in PKEYS}
        gradient_norm = float(sum((p[k].grad.to(torch.float64)**2).sum() for k in PKEYS).sqrt())
        branch_gradient_norm = float(p['ungated_w'].grad.norm())
        optimizer.step()
        assert any(v > 1 for k,v in card.visits.items() if k.startswith('domain:'))
        if rank == 35:
            if gradient35 is None:
                gradient35 = gradients
            else:
                assert gradient35 == gradients
    else:
        with torch.no_grad():
            predicted = card.run(False)
        recorder.point('forward_complete')
        gradients = gradient_norm = branch_gradient_norm = None
    value = predicted.detach().contiguous().numpy().copy()
    assert value.shape == (16,1024) and value.dtype == np.float32 and np.isfinite(value).all() and (value >= 0).all()
    assert {k:branch.tensor_sha(v) for k,v in p.items()} == initial
    assert {k:parent.tensor_sha(v) for k,v in old_p.items()} == old_initial
    assert [branch.tensor_sha(v) for v in [a,c]+list(edge.values())] == fixed
    assert [parent.tensor_sha(v) for v in [old_a,old_c]+list(old_e.values())] == old_fixed
    diagnostics = {k:getattr(card,k) for k in ['domain','selectors','states']}
    assert set(card.domain) == set(card.states) == {str(l) for l in range(16)}
    assert set(card.selectors) == {f'{l}:{s}' for l in range(15) for s in range(3)}
    path = HERE / f'torch_call_{index:02d}_rank{rank:04d}.npy'
    with path.open('xb') as f:
        np.save(f,value,allow_pickle=False)
    if role == 'parent':
        controls[rank] = dict(prediction=pin(path), **diagnostics)
    else:
        control = controls[rank]
        assert pin(path)['sha256'] == control['prediction']['sha256']
        assert all(diagnostics[k] == control[k] for k in diagnostics)
    native = next(r for r in inputs['warm_reference'] if r['rank'] == rank)
    native_value = np.load(native['prediction']['path'], allow_pickle=False)
    difference = value.astype(np.float64) - native_value.astype(np.float64)
    record = dict(index=index,role=role,rank=rank,checkpointed=checkpointed,output=pin(path),
        gradients=gradients,gradient_norm=gradient_norm,branch_gradient_norm=branch_gradient_norm,
        **diagnostics,replay_visits=card.visits,parameters_unchanged=True,
        synthetic_cotangent_only=True,zero_branch_parent_equal=role=='branch',
        native_max_absolute_difference=float(np.max(np.abs(difference))),
        native_final_rms_difference=float(np.sqrt(np.mean(difference[-1]**2))),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if checkpointed:
        del loss
    del predicted,card,weights,native_value,difference,value,args
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    assert ref() is None
    recorder.phase = 'retired'
    record['retirement_memory'] = recorder.point('card_retired')
    assert record['retirement_memory']['after']['rss_bytes'] < 2_000_000_000
    if not checkpointed:
        assert record['peak_rss_bytes'] < 6_500_000_000
    record.update(card_retired=True,wall_s=time.monotonic()-start)
    write(HERE / f'torch_call_{index:02d}_result.json',record)
    records.append(record)
    print(dict(completed_call=index,role=role,rank=rank,wall_s=record['wall_s'],
        gradient_norm=gradient_norm,branch_gradient_norm=branch_gradient_norm,
        peak_rss_bytes=record['peak_rss_bytes'],retired_rss_bytes=record['retirement_memory']['after']['rss_bytes']),flush=True)
recorder.stream.close()
write(HERE / 'torch_qualification.json',dict(passed=True,completed=True,records=records,
    score_labels_accessed=False,native_derivative_certified=False,repeat_gradient_bytes_exact=True,
    all_parameters_unchanged=True,zero_learning_rate_adam_only=True,
    wall_s=time.monotonic()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,completed_calls=len(records)),flush=True)
