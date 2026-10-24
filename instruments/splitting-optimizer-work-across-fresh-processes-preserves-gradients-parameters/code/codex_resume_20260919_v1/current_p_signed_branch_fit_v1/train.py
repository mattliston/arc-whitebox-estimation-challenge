import gc
import math
import os
import resource
import time
import weakref
from fit_common import *

verify(data=True)
assert not (HERE / 'fit_result.json').exists() and not (HERE / 'training_order.json').exists()
resource_floor()
assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'))
import numpy as np
import torch

assert np.__version__ == '1.26.4' and torch.__version__ == '2.7.0'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model, allocation = qualified_modules()
inputs = read(HERE / 'inputs.json')
order = np.random.Generator(np.random.PCG64DXSM(9409212)).permutation(inputs['train']).tolist()
assert len(order) == len(set(order)) == 160 and set(order) == set(inputs['train'])
write(HERE / 'training_order.json', dict(seed=9409212, generator='PCG64DXSM', ranks=order,
    updates=40, batch_networks=4, training_targets_decoded=False))
p, a, c, edge = model.torch_payload(torch, np, True)
initial = {k: model.tensor_sha(v) for k, v in p.items()}
inactive = {k: model.tensor_sha(p[k][0]) for k in model.INACTIVE}
fixed = [model.tensor_sha(v) for v in [a, c] + list(edge.values())]
parameters = [p[k] for k in model.PKEYS]
optimizer = torch.optim.Adam(parameters, lr=5e-4, betas=(.9, .999), eps=1e-8,
    weight_decay=0., foreach=False)
recorder = allocation.Recorder(allocation.Allocator(), HERE / 'allocation_events.jsonl')
Card = allocation.wrapped_card(model, recorder)
began = time.monotonic()
records, updates = [], []


def frozen_check():
    assert all(model.tensor_sha(v) == initial[k] for k, v in p.items() if k not in model.PKEYS)
    assert {k: model.tensor_sha(p[k][0]) for k in model.INACTIVE} == inactive
    assert [model.tensor_sha(v) for v in [a, c] + list(edge.values())] == fixed
    assert all(bool(torch.isfinite(v).all()) for v in p.values())


for update in range(40):
    lr = 5e-5 + .5 * (5e-4 - 5e-5) * (1 + math.cos(math.pi * update / 39))
    optimizer.param_groups[0]['lr'] = lr
    optimizer.zero_grad(set_to_none=True)
    batch = order[4 * update:4 * update + 4]
    for within, rank in enumerate(batch):
        index = update * 4 + within
        print(dict(starting_graph=index, rank=rank, update=update + 1), flush=True)
        start = time.monotonic()
        resource_floor()
        wp, tp = inputs['weights'][str(rank)], inputs['targets'][str(rank)]
        assert rank in inputs['train'] and rank not in inputs['valid']
        check(wp)
        check(tp)
        weights = np.load(wp['path'], allow_pickle=False)
        assert weights.shape == (16, 1024, 1024) and weights.dtype == np.float32
        with np.load(tp['path'], allow_pickle=False) as src:
            target_np = src['mc_post'].copy()
        assert target_np.shape == (16, 1024) and np.isfinite(target_np).all()
        target = torch.tensor(target_np, dtype=torch.float64)
        recorder.call, recorder.phase = index, 'setup'
        recorder.point('training_call_start')
        card = Card(torch, weights, p, a, c, edge)
        card_ref = weakref.ref(card)
        before = {k: model.tensor_sha(v) for k, v in p.items()}
        recorder.phase = 'forward'
        prediction = card.run(True)
        recorder.point('forward_complete')
        forward_s = time.monotonic() - start
        error = prediction.to(torch.float64) - target
        final_mse = error[-1].square().mean()
        earlier_mse = error[:-1].square().mean()
        loss = (final_mse + .05 * earlier_mse) / 2e-8 / 4
        assert bool(torch.isfinite(loss))
        recorder.phase = 'backward'
        loss.backward()
        recorder.point('backward_complete')
        assert all(p[k].grad is not None and bool(torch.isfinite(p[k].grad).all()) for k in model.PKEYS)
        assert all(int(torch.count_nonzero(p[k].grad[0])) == 0 for k in model.INACTIVE)
        assert {k: model.tensor_sha(v) for k, v in p.items()} == before
        frozen_check()
        assert set(card.domain) == {str(l) for l in range(16)}
        assert set(card.states) == {str(l) for l in range(16)}
        assert set(card.selectors) == {f'{l}:{s}' for l in range(15) for s in range(3)}
        assert any(v > 1 for k, v in card.visits.items() if k.startswith('domain:'))
        output = prediction.detach().contiguous().numpy().copy()
        assert output.shape == (16, 1024) and output.dtype == np.float32 and np.isfinite(output).all() and (output >= 0).all()
        output_path = HERE / f'train_{index:03d}_rank{rank:04d}.npy'
        assert not output_path.exists()
        np.save(output_path, output, allow_pickle=False)
        record = dict(index=index, rank=rank, update=update + 1, within_batch=within,
            prediction=pin(output_path), weights=wp, target=tp, forward_s=forward_s,
            final_mse=float(final_mse.detach()), earlier_mse=float(earlier_mse.detach()),
            batch_scaled_loss=float(loss.detach()), parameter_hashes_during_graph=before,
            domain=card.domain, selectors=card.selectors, states=card.states, visits=card.visits,
            fixed_parameters_unchanged=True, validation_targets_accessed=False,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        del card, weights, target, target_np, prediction, output, error, final_mse, earlier_mse, loss
        gc.collect()
        assert card_ref() is None
        recorder.phase = 'retired'
        record['retirement_memory'] = recorder.point('training_card_retired')
        assert record['retirement_memory']['after']['rss_bytes'] < 2_000_000_000
        record['wall_s'] = time.monotonic() - start
        record['card_retired'] = True
        write(HERE / f'train_{index:03d}_result.json', record)
        records.append(dict(rank=rank, index=index, record=pin(HERE / f'train_{index:03d}_result.json'),
            final_mse=record['final_mse'], earlier_mse=record['earlier_mse'], wall_s=record['wall_s']))
        print(dict(completed_graph=index, rank=rank, final_mse=record['final_mse'], wall_s=record['wall_s'],
            peak_rss=record['peak_rss_bytes'], elapsed_s=time.monotonic() - began), flush=True)
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True, foreach=False))
    optimizer.step()
    frozen_check()
    assert len(optimizer.state) == len(parameters)
    assert all(all(bool(torch.isfinite(v).all()) for v in s.values()) for s in optimizer.state.values())
    update_record = dict(update=update + 1, ranks=batch, learning_rate=lr,
        pre_clip_gradient_norm=norm, fixed_parameters_unchanged=True,
        parameter_hashes={k: model.tensor_sha(v) for k, v in p.items()},
        elapsed_s=time.monotonic() - began, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    path = HERE / f'update_{update + 1:02d}.json'
    write(path, update_record)
    updates.append(pin(path))
    print(dict(completed_update=update + 1, gradient_norm=norm, learning_rate=lr,
        elapsed_s=update_record['elapsed_s']), flush=True)

assert len(records) == 160 and len(updates) == 40
frozen_check()
payload = {k: v.detach().contiguous().numpy().copy() for k, v in p.items()}
payload_path = HERE / 'terminal_payload.npz'
assert not payload_path.exists()
np.savez(payload_path, **payload)
changed = [k for k in model.PKEYS if model.tensor_sha(p[k]) != initial[k]]
assert changed
optimizer.zero_grad(set_to_none=True)
optimizer.state.clear()
parameters.clear()
gc.collect()
recorder.point('fit_complete')
recorder.stream.close()
result = dict(completed=True, passed=True, model_fitted=True, training_graphs=160, updates=40,
    records=records, update_receipts=updates, terminal_payload=pin(payload_path), epoch=1,
    initial_parameter_hashes=initial, terminal_parameter_hashes={k: model.tensor_sha(v) for k, v in p.items()},
    changed_trainable_keys=changed, fixed_parameters_unchanged=True, inactive_parameters_unchanged=True,
    validation_targets_accessed=False, training_targets_accessed=True, checkpoint_selection=False,
    native_derivative_certified=False, native_score_measured=False, candidate_created=False,
    training_order=pin(HERE / 'training_order.json'), source_seal=pin(HERE / 'sources_sealed.json'),
    wall_s=time.monotonic() - began, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
verify()
resource_floor()
write(HERE / 'fit_result.json', result)
print(dict(completed=True, fit_result=pin(HERE / 'fit_result.json'), wall_s=result['wall_s'],
    peak_rss_bytes=result['peak_rss_bytes']), flush=True)
