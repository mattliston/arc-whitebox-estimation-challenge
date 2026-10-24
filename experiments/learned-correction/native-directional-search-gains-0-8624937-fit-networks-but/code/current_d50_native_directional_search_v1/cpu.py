"""Inert serial CPU-function qualification and ONE unqualified search proposal.

The frozen worker is a proposal engine, never a certified native derivative.
Only --run admits numerical imports; only propose, after both new prerequisites,
may decode the four fitting TRAIN targets. No check/VALID targets are read.
"""
import argparse
import gc
import importlib.util
import math
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
PROTOCOL_SHA = '93ec25b16e960758a38cc05e5035cbe426aca19b96b5bd3ebb1c718b8895e2ce'
WORKER_SHA = '16e4762e49d94401041946a0cebc4b69dbfe9ae13c7c875e1834eeb73f3a46d1'
HY_CPU_SHA = 'acfe54dfaffc19ad1d1c2c56c2349750ceff0cf4a587b72638827c7767abea0e'
CO_CPU_SHA = '6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607'
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
INACTIVE = ('head', 'head_b', 'gain')
H = 5e-4


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Module specification unavailable')
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def bind_inventory(s, mapping):
    """Accept either archived full bindings or flat path-to-SHA inventories."""
    for path, item in mapping.items():
        digest = item if isinstance(item, str) else item['sha256']
        if isinstance(item, dict):
            s.c.require(Path(item['path']).resolve() == Path(path).resolve(),
                        'Archived binding path differs from its key')
        group = 'source_bindings' if Path(path).suffix == '.py' else 'input_bindings'
        s.pin(path, digest, group=group)


def cpu_reference(s):
    c = s.c
    path = c.NEW / 'current_d50_hybrid_cp_qualification_v1/stage_cpu_gradient_v1/result.json'
    s.pin(path, HY_CPU_SHA)
    s.pin(path.parent / 'completion.json')
    record, seal = c.read(path), c.read(path.parent / 'completion.json')
    c.require(record['schema'] == 'current_d50_hybrid_cp_qualification_v1'
              and record['stage'] == 'cpu_gradient' and record['passed']
              and record['completed'] and record['sources_unchanged']
              and seal['passed'] and seal['completed'] and seal['result_sha256'] == HY_CPU_SHA,
              'Completed frozen CPU-gradient reference required')
    c.require(record['no_fit'] and record['parameter_updates'] == 0
              and not record['alltruth_accessed'] and not record['training_targets_accessed']
              and not record['validation_targets_accessed']
              and record['probe_owner_sha256'] == CO_CPU_SHA,
              'Synthetic reference lineage/target scope')
    for group in ('source_bindings', 'artifacts', 'dependencies', 'weight_bindings'):
        bind_inventory(s, record[group])
    s.record['cpu_reference_sha256'] = HY_CPU_SHA
    return record


def original_payload(np, s):
    c = s.c
    s.pin(c.P0_PATH, c.P0_SHA)
    with np.load(c.P0_PATH, allow_pickle=False) as src:
        values = {key: src[key].copy() for key in src.files}
    c.require(set(PKEYS).issubset(values), 'Complete original P payload')
    c.require(all(type(x) is np.ndarray and x.dtype == np.float32 and np.isfinite(x).all()
                  for x in values.values()), 'Finite original full FP32 payload')
    c.require(sum(values[key].size for key in PKEYS) == 45200
              and sum(values[key][0].size for key in INACTIVE) == 66,
              '45200 allocated/45134 active P coordinates')
    return values


def same_bundle(np, c, left, right):
    return (set(left) == set(right)
            and all(left[k].shape == right[k].shape and left[k].dtype == right[k].dtype
                    and left[k].tobytes(order='C') == right[k].tobytes(order='C') for k in left))


def domain(c, diagnostics, replay):
    c.require(diagnostics['frozen_cstar'] and not diagnostics['alltruth_accessed']
              and diagnostics['inverse_source_byte_exact']
              and diagnostics['allocated_parameters'] == 45200
              and diagnostics['active_parameters'] == 45134
              and diagnostics['moment_dtype'] == 'float64'
              and diagnostics['moment_device'] == 'cpu' and diagnostics['parameter_device'] == 'cpu'
              and diagnostics['carrier_product_device'] == 'cpu'
              and diagnostics['hybrid_enabled'] is False
              and diagnostics['offloaded_products'] == 0
              and diagnostics['transferred_bytes'] == 0 and diagnostics['training'],
              'Exact CPU proposal-card semantics')
    rows = diagnostics['variance_domain']
    c.require(set(rows) == {str(l) for l in range(16)}
              and all(x['finite'] and math.isfinite(x['minimum'])
                      and x['minimum'] >= 1e-10 and x['below_floor_count'] == 0
                      for x in rows.values()), 'All16 finite prevariance/no-floor domains')
    expected = {f'{l}:{slot}' for l in range(16) for slot in range(2 if l == 15 else 3)}
    cells = diagnostics['selection_cells']
    c.require(set(cells) == expected, 'All47 original selector cells')
    c.require(all(sorted(row['ordered_ids']) == row['membership']
                  and len(set(row['ordered_ids'])) == row['count'] == len(row['ordered_ids'])
                  for row in cells.values()), 'Selector order/membership integrity')
    if replay:
        c.require(max(diagnostics['variance_visits'].values()) >= 2
                  and max(diagnostics['selection_visits'].values()) >= 2,
                  'Actual checkpoint replay required')


def save_packet(np, s, worker, packet, ordinal):
    """Seal returned measurements before applying stage-specific comparisons."""
    c = s.c
    row = {key: value for key, value in packet.items() if key not in ('prediction', 'gradients')}
    prefix = f'packet_{ordinal:02d}_rank{packet["rank"]:04d}'
    if 'prediction' in packet:
        row['prediction'] = s.save(np, prefix + '_prediction.npy', packet['prediction'])
    if packet.get('gradients'):
        row['gradient'] = s.bundle(np, prefix + '_gradient.npz', packet['gradients'])
    s.record['tasks'].append(row)
    s.attempt(prefix + '.json', row)
    c.require(packet['passed'] and packet['completed'] and packet['task_state_released'],
              'Worker failed or did not retire its task')
    c.require(packet['pid'] == s.record['worker']['pid']
              and packet['versions'] == s.record['worker']['versions']
              and packet['parameter_updates'] == 0 and packet['mps_used'] is False
              and packet['target_files_accessed'] is False
              and packet['peak_rss_bytes'] < 11_000_000_000,
              'Serial CPU worker/frame/resource contract')
    bind_inventory(s, packet['source_bindings'])
    bind_inventory(s, packet['weight_bindings'])
    c.require(packet['source_bindings'] == s.record['worker']['source_bindings'],
              'Worker source inventory changed')
    domain(c, packet['forward_diagnostics'], replay=False)
    domain(c, packet['diagnostics'], replay=True)
    c.require(packet['forward_diagnostics']['selection_cells'] == packet['diagnostics']['selection_cells']
              and packet['forward_diagnostics']['variance_domain'] == packet['diagnostics']['variance_domain'],
              'Checkpoint forward/backward state chart changed')
    gradients = packet['gradients']
    c.require(set(gradients) == set(PKEYS) and sum(x.size for x in gradients.values()) == 45200
              and all(x.dtype == np.float32 and np.isfinite(x).all() for x in gradients.values()),
              'All45200 finite FP32 gradients')
    c.require(packet['inactive_gradients_zero']
              and all(bool((gradients[k][0] == 0).all()) for k in INACTIVE), 'Inactive gradients')
    c.require(worker.array_metadata(packet['prediction']) == packet['prediction_metadata']
              and {key: worker.array_metadata(value) for key, value in gradients.items()}
              == packet['gradient_metadata'], 'Returned array metadata changed')
    c.require(packet['prediction'].shape == (16, 1024)
              and packet['prediction'].dtype == np.float32
              and np.isfinite(packet['prediction']).all() and (packet['prediction'] >= 0).all(),
              'Finite nonnegative all16 served FP32 outputs')
    return row


def qualify(np, s, worker, original, reference):
    c = s.c
    archived = s.load_bundle(np, reference['payloads']['null'])
    c.require(same_bundle(np, c, original, archived), 'Saved null and original payload array bytes')
    s.record['payloads'] = {'baseline': s.bundle(np, 'payload_baseline.npz', original)}
    expected_payload, _ = worker.payload_metadata(original)
    for ordinal, rank in enumerate((35, 100)):
        cotangent = s.load_array(np, reference['cotangents'][str(rank)])
        task = dict(id=f'qualify_{rank}', rank=rank, mode='synthetic', payload=original,
                    cotangent=cotangent, loss_scale=1.)
        packet = worker.evaluate(task)
        row = save_packet(np, s, worker, packet, ordinal)
        old = reference['gradients'][str(rank)]
        prediction = s.load_array(np, old['prediction'])
        gradients = s.load_bundle(np, old['gradient'])
        comparison = dict(rank=rank,
            prediction_byte_exact=packet['prediction'].dtype == prediction.dtype
                and packet['prediction'].shape == prediction.shape
                and packet['prediction'].tobytes() == prediction.tobytes(),
            per_key_gradient_byte_exact={key: packet['gradients'][key].dtype == gradients[key].dtype
                and packet['gradients'][key].shape == gradients[key].shape
                and packet['gradients'][key].tobytes() == gradients[key].tobytes() for key in PKEYS},
            forward_diagnostics_exact=packet['forward_diagnostics'] == old['forward_diagnostics'],
            backward_diagnostics_exact=packet['diagnostics'] == old['diagnostics'],
            synthetic_loss_exact=packet['synthetic_loss'] == old['synthetic_loss'],
            payload_identity_exact=packet['payload_sha256'] == expected_payload)
        row['qualification_comparison'] = comparison
        s.attempt(f'comparison_rank{rank:04d}.json', comparison)
        c.require(comparison['prediction_byte_exact'] and all(comparison['per_key_gradient_byte_exact'].values())
                  and comparison['forward_diagnostics_exact'] and comparison['backward_diagnostics_exact']
                  and comparison['synthetic_loss_exact'] and comparison['payload_identity_exact'],
                  'Frozen CPU function did not reproduce all reference bytes/state/scalar')
        del packet, task, cotangent, gradients, prediction
        gc.collect()
        worker.guard()
    s.record.update(qualification_passed=True, qualified_ranks=[35, 100],
                    qualified_cpu_function_only=True, parameter_updates=0,
                    all_task_states_released=True, no_fit=True)


def proposal_prerequisites(s, args):
    c = s.c
    qualified = s.receipt(HERE / 'qualification_v1', args.qualification_result_sha256, 'qualify')
    baseline = s.receipt(HERE / 'native_baseline_v1', args.baseline_result_sha256, 'native', 'baseline')
    c.require(qualified['qualification_passed'] and qualified['qualified_cpu_function_only']
              and qualified['qualified_ranks'] == [35, 100] and qualified['no_fit']
              and not qualified['training_targets_accessed'] and qualified['parameter_updates'] == 0
              and qualified['all_task_states_released'] and not qualified['native_gradient_qualified'],
              'New target-free CPU qualification required')
    c.require(all(baseline[key] for key in ('baseline_reference_passed', 'all_domain_checks_passed',
                  'all_selector_checks_passed', 'repeat_byte_exact', 'own_complete_bills'))
              and baseline['initialization_flops'] == 1383913786
              and not baseline['truth_accessed'] and not baseline['alltruth_accessed']
              and [row['rank'] for row in baseline['trials']] == [35, 36, 38, 40, 41, 43, 44, 45, 100, 35],
              'New complete native baseline fidelity/resources/roster prerequisite')
    s.record['prerequisites'] = dict(qualification_result_sha256=args.qualification_result_sha256,
                                    baseline_result_sha256=args.baseline_result_sha256)
    return qualified


def proposal(np, torch, s, worker, original, qualified):
    c = s.c
    c.require(tuple(c.FIT) == (35, 36, 38, 40) and tuple(c.CHECK) == (41, 43, 44, 45),
              'Fixed fitting/proposal-check split')
    c.require(same_bundle(np, c, original, s.load_bundle(np, qualified['payloads']['baseline'])),
              'Qualified baseline full payload changed')
    aggregate = {key: torch.zeros_like(torch.from_numpy(original[key])) for key in PKEYS}
    expected_payload, _ = worker.payload_metadata(original)
    s.record.update(fit_ranks=list(c.FIT), check_ranks=list(c.CHECK), target_ranks_accessed=[],
                    loss_convention='CPU served FP32; FP64 target/subtraction/reduction; '
                                    '(final_MSE+.05*earlier_MSE)/2e-8; per-net scale .25',
                    surrogate_loss_is_not_native_objective=True, no_fit=False)
    # The four metadata bindings and all prerequisite seals precede target decoding.
    targets = {rank: s.target(rank, c.FIT) for rank in c.FIT}
    s.attempt('before_fitting_target_access.json', dict(prerequisites=s.record['prerequisites'],
              target_bindings=dict(s.record['target_bindings']), ranks=list(c.FIT),
              check_targets_accessed=False, validation_targets_accessed=False))
    for ordinal, rank in enumerate(c.FIT):
        s.pin(targets[rank]['path'], targets[rank]['sha256'], group='target_bindings')
        s.record['training_targets_accessed'] = True
        s.record['truth_accessed'] = True
        s.record['target_ranks_accessed'].append(rank)
        with np.load(targets[rank]['path'], allow_pickle=False) as src:
            target = np.asarray(src['mc_post'], dtype=np.float64).copy()
        c.require(target.shape == (16, 1024) and np.isfinite(target).all(), 'Finite fixed TRAIN target')
        task = dict(id=f'propose_{rank}', rank=rank, mode='train', payload=original,
                    target=target, loss_scale=.25)
        packet = worker.evaluate(task)
        save_packet(np, s, worker, packet, ordinal)
        c.require(packet['payload_sha256'] == expected_payload and packet['loss_scale'] == .25,
                  'One unchanged P0 and .25-scaled gradient for each fitting net')
        c.require(packet['target_metadata'] == worker.array_metadata(target), 'Worker target identity')
        for key in PKEYS:
            c.require(packet['gradients'][key].shape == original[key].shape, 'Gradient parameter shape')
            aggregate[key].add_(torch.from_numpy(packet['gradients'][key]))
        s.pin(targets[rank]['path'], targets[rank]['sha256'], group='target_bindings')
        del packet, task, target
        gc.collect()
        worker.guard()
    c.require(s.record['target_ranks_accessed'] == list(c.FIT), 'Only fixed fitting truths decoded')
    before_clip = {key: value.numpy().copy() for key, value in aggregate.items()}
    s.record['aggregate_gradient'] = s.bundle(np, 'aggregate_gradient_fp32.npz', before_clip)
    parameters = {key: torch.tensor(original[key], dtype=torch.float32, requires_grad=True) for key in PKEYS}
    ordered = [parameters[key] for key in PKEYS]
    for key in PKEYS:
        parameters[key].grad = aggregate[key].clone()
    c.require(all(bool((parameters[key].grad[0] == 0).all()) for key in INACTIVE), 'Inactive aggregate gradient')
    optimizer = torch.optim.Adam(ordered, lr=H, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., foreach=False)
    norm = float(torch.nn.utils.clip_grad_norm_(ordered, 5., error_if_nonfinite=True, foreach=False))
    c.require(math.isfinite(norm), 'Finite global gradient norm')
    s.record['clipped_gradient'] = s.bundle(np, 'clipped_gradient_fp32.npz',
                                         {key: parameters[key].grad.numpy().copy() for key in PKEYS})
    optimizer.step()
    c.require(all(float(optimizer.state[value]['step']) == 1. for value in ordered), 'Exactly one Adam step')
    plus = {key: parameters[key].detach().numpy().copy() if key in parameters else value.copy()
            for key, value in original.items()}
    c.require(all(np.isfinite(x).all() and x.dtype == np.float32 for x in plus.values()), 'Finite FP32 endpoint')
    c.require(all(plus[key][0].tobytes() == original[key][0].tobytes() for key in INACTIVE),
              'Adam changed an inactive layer0 coordinate')
    difference = {key: plus[key].astype(np.float64) - value.astype(np.float64)
                  for key, value in original.items()}

    def curve(t):
        values = {key: (value.astype(np.float64) + (t / H) * difference[key]).astype(np.float32)
                  if key in PKEYS else value.copy() for key, value in original.items()}
        for key in INACTIVE:
            values[key][0] = original[key][0]
        return values

    replay, minus = curve(H), curve(-H)
    c.require(same_bundle(np, c, replay, plus), 'Declared FP64 difference curve does not recover Adam endpoint')
    c.require(same_bundle(np, c, curve(0.), original), 'Declared curve does not recover original payload bytes')
    for values in (plus, minus):
        c.require(all(x.dtype == np.float32 and np.isfinite(x).all() for x in values.values()), 'Finite payload curve')
        c.require(all(values[key].tobytes() == value.tobytes() for key, value in original.items() if key not in PKEYS)
                  and all(values[key][0].tobytes() == original[key][0].tobytes() for key in INACTIVE),
                  'Frozen payload fields/unused rows changed')
    c.require(all(bool((difference[key] == 0).all()) for key in difference if key not in PKEYS)
              and all(bool((difference[key][0] == 0).all()) for key in INACTIVE), 'Frozen difference coordinates')
    metrics = {}
    for key in PKEYS:
        baseline_rms = float(np.sqrt(np.mean(original[key].astype(np.float64) ** 2)))
        metrics[key] = dict(baseline_rms=baseline_rms)
        for name, values in (('plus', plus), ('minus', minus)):
            delta = values[key].astype(np.float64) - original[key].astype(np.float64)
            rms = float(np.sqrt(np.mean(delta ** 2)))
            metrics[key][name] = dict(maximum_absolute=float(np.max(np.abs(delta))), rms=rms,
                rms_relative_to_baseline=rms / baseline_rms if baseline_rms > 0 else None,
                baseline_rms_zero=baseline_rms == 0, changed_coordinates=int(np.count_nonzero(delta)))
    s.record['payloads'] = {name: s.bundle(np, 'payload_' + name + '.npz', values)
                           for name, values in (('baseline', original), ('plus', plus), ('minus', minus))}
    s.record['difference64'] = s.bundle(np, 'difference64.npz', difference)
    s.record['direction64'] = s.bundle(np, 'direction64.npz', {key: value / H for key, value in difference.items()})
    s.record.update(parameter_updates=1, adam_steps=1, trainable_keys=list(PKEYS),
                    allocated_parameters=45200, active_parameters=45134, gradient_norm_before_clip=norm,
                    optimizer=dict(name='Adam', lr=H, betas=[.9, .999], eps=1e-8,
                                   weight_decay=0., foreach=False, global_norm_cap=5.),
                    curve_h=H, curve_endpoint_byte_exact=True, frozen_payload_fields_byte_exact=True,
                    payload_change_metrics=metrics, all_task_states_released=True,
                    proposal_sealed=True, surrogate_only=True, no_native_step_selection=True)
    s.attempt('proposal_seal.json', dict(payloads=s.record['payloads'], difference64=s.record['difference64'],
              direction64=s.record['direction64'], curve_h=H, parameter_updates=1,
              check_targets_accessed=False, validation_targets_accessed=False,
              native_gradient_qualified=False, native_objective_measured=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--stage', choices=('qualify', 'propose'))
    parser.add_argument('--admission-sha256')
    parser.add_argument('--qualification-result-sha256')
    parser.add_argument('--baseline-result-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    required = [args.admission_sha256]
    if args.stage == 'propose':
        required.extend((args.qualification_result_sha256, args.baseline_result_sha256))
    if args.stage is None or any(not isinstance(x, str) or len(x) != 64 for x in required):
        parser.error('Explicit stage, reviewed admission and prerequisite digests required')
    if args.stage == 'qualify' and (args.qualification_result_sha256 or args.baseline_result_sha256):
        parser.error('Qualification has no numerical/target prerequisite overrides')
    c = module(HERE / 'common.py', 'native_directional_common_cpu')
    s = c.Stage(__file__, args.admission_sha256, args.stage,
                'qualification_v1' if args.stage == 'qualify' else 'proposal_v1')
    s.c = c
    try:
        s.environment('AP')
        c.require(c.PROTOCOL_SHA == PROTOCOL_SHA and c.WORKER_SHA == WORKER_SHA, 'Frozen protocol/worker contract')
        s.pin(c.WORKER_PATH, WORKER_SHA, group='source_bindings')
        s.pin(c.P0_PATH, c.P0_SHA)
        s.record.update(tasks=[], payloads={}, target_bindings={}, training_targets_accessed=False,
                        check_targets_accessed=False, validation_targets_accessed=False,
                        native_gradient_qualified=False, native_objective_measured=False,
                        native_score_measured=False, frozen_cstar=True, mps_used=False,
                        all_task_states_released=False, parameter_updates=0, truth_accessed=False)
        reference = cpu_reference(s) if args.stage == 'qualify' else None
        qualified = proposal_prerequisites(s, args) if args.stage == 'propose' else None
        worker = c.module(c.WORKER_PATH, 'native_directional_frozen_cpu_worker')
        bind_inventory(s, worker.authenticate())
        c.require(not any(name in sys.modules for name in ('numpy', 'torch', 'threadpoolctl', 'flopscope')),
                  'Complete source/reference admission before numerical imports')
        metadata = worker.initialize(WORKER_SHA)
        s.record['worker'] = metadata
        c.require(tuple(metadata['PKEYS']) == PKEYS and metadata['parameter_count'] == 45200
                  and metadata['versions'] == dict(numpy='1.26.4', torch='2.7.0')
                  and metadata['mps_used'] is False, 'Frozen CPU worker metadata')
        bind_inventory(s, metadata['source_bindings'])
        import numpy as np
        import torch
        original = original_payload(np, s)
        s.record['original_payload_sha256'] = c.P0_SHA
        if args.stage == 'qualify':
            qualify(np, s, worker, original, reference)
        else:
            proposal(np, torch, s, worker, original, qualified)
        gc.collect()
        s.record['peak_rss_bytes'] = worker.guard()
        s.finish(passed=True, scientific=False)
    except BaseException:
        s.record.update(passed=False, completed=False, hashes_unchanged=False,
                        error=traceback.format_exc())
    finally:
        exit_code = s.close()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
