"""One target-free canonical35 CPU forward of the shared384 proposal model."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
NAMES = ['forward_native_y.py','watch_forward_native_y.py','model.py','shared_operator.py','physical_alignment.py','rangefinder_alignment.py','differentiable_rangefinder.py','RANGEFINDER_FULL_FORWARD_PROTOCOL.md','forward_inputs.json','rangefinder_full_inputs.json']

NEW, P2 = HERE.parent, HERE.parent.parent
GAIN = NEW/'current_comp_shared_gain_slice_v1'
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
WEIGHTS = STORE/'rows/observer_fit/0035/weights.npy'
REFERENCE = GAIN/'native_panel_v1/call_00_rank35.npy'
DIAGNOSTICS = GAIN/'native_panel_v1/call_00_diagnostics.json'
LIMITS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def sealed(folder, pins):
    paths = [folder/name for name in ('result.json','completion.json','finalization.json')]
    require(all(str(p) in pins and sha(p)==pins[str(p)] for p in paths), 'Exact prerequisite file pins')
    r, c, f = map(read, paths)
    require(all(x['passed'] is True and x['completed'] is True for x in (r,c,f)) and
            c['result_sha256']==f['result_sha256']==pins[str(paths[0])] and
            f['completion_sha256']==pins[str(paths[1])], 'Complete prerequisite three seals')
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pins', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    began = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Whole120s forward')))
    signal.alarm(120)
    output = Path(args.output).resolve()
    require(output == HERE/'cpu_native_y_forward_v1' and not output.exists(), 'One fresh fixed output')
    output.mkdir()
    result = dict(passed=False, completed=False, full_model_calls=0, fits=0, truth_accessed=False,
                  native_gradient_certified=False, native_score_measured=False)
    try:
        pins = read(args.pins)
        require(set(pins)==set(NAMES),'Exact ten reviewed native-Y forward owners')
        require(all(sha(HERE/name)==h for name,h in pins.items()), 'Reviewed forward source pins')
        inputs = read(HERE/'forward_inputs.json')
        required_inputs = {
            WEIGHTS: '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2',
            WEIGHTS.parent/'row_manifest.json': '5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809',
            STORE/'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
        }
        require(all(inputs.get(str(p))==h for p,h in required_inputs.items()),
                'Mandatory exact canonical35 weight/row/store identity')
        require(all(Path(p).resolve().is_relative_to(P2) and 'truthpost' not in Path(p).parts
                    and sha(p)==h for p,h in inputs.items()), 'Pinned target-free prerequisite inputs')
        ar = sealed(HERE/'operator_analytic_v1', inputs)
        aw = sealed(HERE/'operator_analytic_watch_v1', inputs)
        require(all(aw['child_'+kind+'_sha256']==inputs[str(HERE/'operator_analytic_v1'/(kind+'.json'))]
                    for kind in ('result','completion','finalization')),
                'Actual analytic worker joined to its independent watcher')
        require(ar['qualification_scope']=='independent_analytic_JVP_and_checkpoint_not_original_FD'
                and ar['original_finite_difference_passed'] is False and
                ar['analytic']['absolute_error']<=1e-8 and ar['analytic']['relative_error']<=1e-5 and
                ar['gradient']['checkpoint_gradients_byte_exact'] and
                aw['whole_owned_tree_retired'] and aw['maximum_swap_growth_bytes']==0,
                'Independent analytic and checkpoint prerequisite, old FD still failed')
        nr = sealed(GAIN/'native_panel_v1', inputs)
        trial = nr['trials'][0]
        require(nr['candidate_sha256']=='c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
                and trial['rank']==35 and trial['call']==0 and trial['output']==str(REFERENCE)
                and trial['flops']==403400296797 and
                trial['output_bytes_sha256']=='84b1f1dc640209d009483d1fbacb0424130fb59eb32bb79b8ee88fa6a0eb9819'
                and nr['artifacts'][str(REFERENCE)]==inputs[str(REFERENCE)]
                and nr['artifacts'][str(DIAGNOSTICS)]==inputs[str(DIAGNOSTICS)], 'Exact current-c2 native control')
        extra = read(HERE/'rangefinder_full_inputs.json')
        require(all(Path(p).resolve().is_relative_to(HERE) and sha(p)==h for p,h in extra.items()),
                'Exact fixed local VJP inputs and positive receipt pins')
        vr = sealed(HERE/'rangefinder_vjp_v1', extra)
        vw = sealed(HERE/'rangefinder_vjp_watch_v1', extra)
        require(vr['unit_cases']==3 and vr['rangefinder_real_algebra_VJP_qualified']
                and vr['all_owners_retired'] and vw['whole_owned_tree_retired']
                and vw['maximum_swap_growth_bytes']==0 and
                all(vw['child_'+kind+'_sha256']==extra[str(HERE/'rangefinder_vjp_v1'/(kind+'.json'))]
                    for kind in ('result','completion','finalization')),
                'Actual qualified VJP/checkpoint source, numerical and independent retirement joins')
        require(vr['diagnostic_sha256']==extra[str(HERE/'rangefinder_vjp_v1/diagnostics.json')],
                'Actual positive VJP diagnostic bytes')
        require(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1' for k in LIMITS),
                'Actual nice10/six BLAS limits1')
        require(not {'torch','numpy','flopscope','threadpoolctl'} & set(sys.modules), 'Inert admission')
        spec = importlib.util.spec_from_file_location('current_shared_native_y_alignment', HERE/'rangefinder_alignment.py')
        alignment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alignment)
        model = alignment.model(pins['rangefinder_alignment.py'])
        bindings = model.authenticate(pins['shared_operator.py'])
        result['rangefinder_alignment_source_proof'] = model.rangefinder_alignment_source_proof
        bridge, prepared = model.native_bridge, model.native_prepared
        selected = bridge.select_backend(prepared)
        require(not {'torch','numpy','flopscope','threadpoolctl'} & set(sys.modules), 'All current head admission inert')
        import numpy as np
        import torch
        from threadpoolctl import threadpool_info, threadpool_limits
        thread_limits = threadpool_limits(limits=1)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        attestation = bridge.attest(prepared,np,torch,threadpool_info)
        workspace = bridge.build_workspace(prepared,np,attestation)
        weights = np.load(WEIGHTS, allow_pickle=False)
        reference = np.load(REFERENCE, allow_pickle=False)
        require(weights.shape==(16,1024,1024) and weights.dtype==np.float32 and
                reference.shape==(16,1024) and reference.dtype==np.float32 and
                np.isfinite(weights).all() and np.isfinite(reference).all(), 'Actual input shapes/dtypes/finite')
        require(hashlib.sha256(reference.tobytes()).hexdigest()==trial['output_bytes_sha256'], 'Actual reference bytes')
        P, A = model.frozen('cpu', trainable_P=False)
        require(not A.requires_grad and all(not v.requires_grad for v in P.values()), 'No live trainable leaves')
        initial = {k:hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for k,v in P.items()}
        initial_A = hashlib.sha256(A.detach().numpy().tobytes()).hexdigest()
        card = model.card(35, P=P, A=A, device='cpu', weights=weights,
                          native_workspace=workspace,native_attestation=attestation,numpy_api=np)
        initial_cstar = hashlib.sha256(card.coefficients.detach().numpy().tobytes()).hexdigest()
        started = time.monotonic()
        with torch.no_grad():
            raw, trace = card.run(training=False)
            served = model._JOINT._CO.served(raw)
        result['full_model_calls'] = 1
        result['model_elapsed_s'] = time.monotonic()-started
        predicted = served.detach().cpu().contiguous().numpy().copy()
        require(predicted.shape==(16,1024) and predicted.dtype==np.float32 and
                np.isfinite(predicted).all() and (predicted>=0).all(), 'Finite correctly served Phase2 output')
        diagnostics = card.diagnostics()
        cells = {f'{l}:{s}' for l in range(16) for s in range(2 if l==15 else 3)}
        require(set(diagnostics['variance_domain'])=={str(l) for l in range(16)} and
                set(diagnostics['selection_cells'])==cells and
                set(diagnostics['shared_cells'])=={str(l) for l in range(16)}, 'All16 domains/shared states and47selectors')
        require(all(v['finite'] and v['minimum']>=1e-10 and v['below_floor_count']==0
                    for v in diagnostics['variance_domain'].values()), 'Every unfloored variance domain')
        for record in diagnostics['selection_cells'].values():
            require(sorted(record['ordered_ids'])==record['membership'] and
                    len(set(record['ordered_ids']))==record['count'], 'Well-formed actual ordered selector')
        native_d = read(DIAGNOSTICS)['domains']
        comparison = {key:dict(membership_equal=diagnostics['selection_cells'][key]['membership']==native_d['selection_cells'][key]['membership'],
                              order_equal=diagnostics['selection_cells'][key]['ordered_ids']==native_d['selection_cells'][key]['ordered_ids']) for key in sorted(cells)}
        difference = predicted.astype(np.float64)-reference.astype(np.float64)
        result['fidelity'] = dict(max_abs=float(np.max(np.abs(difference))),
            per_layer_max_abs=np.max(np.abs(difference),axis=1).tolist(),
            per_layer_rms=np.sqrt(np.mean(difference*difference,axis=1)).tolist(),
            native_selector_comparisons=comparison, max_abs_limit=1e-5)
        with (output/'served.npy').open('xb') as stream:
            np.save(stream,predicted,allow_pickle=False)
        write(output/'diagnostics.json',diagnostics)
        result['artifacts'] = {str(p):sha(p) for p in (output/'served.npy',output/'diagnostics.json')}
        require(all(hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()==initial[k] for k,v in P.items())
                and hashlib.sha256(A.detach().numpy().tobytes()).hexdigest()==initial_A and
                hashlib.sha256(card.coefficients.detach().numpy().tobytes()).hexdigest()==initial_cstar,
                'Every P/A/cstar byte unchanged')
        refs = [weakref.ref(card),weakref.ref(raw),weakref.ref(served),weakref.ref(card.coefficients)] + [weakref.ref(v) for v in P.values()] + [weakref.ref(A)]
        refs += [weakref.ref(workspace),weakref.ref(card._native_rangefinder)]
        del card, raw, served, P, A, weights, predicted, reference, difference
        workspace.close();del workspace
        gc.collect()
        require(all(ref() is None for ref in refs), 'Card/output/parameter owners retired')
        final_attestation=bridge.attest(prepared,np,torch,threadpool_info)
        require(all(final_attestation['source_bindings'].get(p)==h for p,h in attestation['source_bindings'].items()),
                'Previously imported numerical owners unchanged')
        require(all(sha(p)==h for p,h in bindings.items()) and all(sha(p)==h for p,h in inputs.items())
                and all(sha(p)==h for p,h in extra.items())
                and all(sha(p)==h for p,h in final_attestation['source_bindings'].items())
                and all(sha(HERE/k)==h for k,h in pins.items()), 'Every source/input/backend unchanged')
        result.update(backend_selection=selected,backend_attestation=final_attestation,
                      VJP_input_bindings=extra,complete_source_upper=2_000_000_000_000,
                      actual_deployment_bill_claimed=False)
        thread_limits.restore_original_limits()
        result.update(completed=True, source_bindings=bindings, own_source_bindings=pins,
                      input_bindings=inputs, all_owners_retired=True, parameters_unchanged=True)
        result['passed'] = result['fidelity']['max_abs']<=1e-5
        result['fidelity_passed'] = result['passed']
        print(json.dumps(dict(event='forward_complete',model_elapsed_s=result['model_elapsed_s'],
                             max_abs=result['fidelity']['max_abs'],passed=result['passed'])),flush=True)
    except BaseException:
        result['error'] = traceback.format_exc()
    result['elapsed_s'] = time.monotonic()-began
    result['maximum_self_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if result['maximum_self_rss_bytes']>=8_000_000_000 or result['elapsed_s']>=120:
        result.update(passed=False, resource_error='Whole120s/8GB host ceiling')
    write(output/'result.json',result)
    rh = sha(output/'result.json')
    write(output/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=rh))
    write(output/'finalization.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=rh,
        completion_sha256=sha(output/'completion.json'),elapsed_s=time.monotonic()-began))
    signal.alarm(0)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())


