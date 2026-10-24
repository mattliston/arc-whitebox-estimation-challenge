"""One-shot, target-free hybrid qualification. Explicit --run required."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
CO = HERE.parent / 'current_bd_p_coadaptation_v1'
STAGES = ('mps_forward', 'cpu_gradient', 'mps_gradient')
HELPER_SHA = '1a3df2687ed3bef6ffcfc6f690f3cfe8665ad04aae4da94012c608d4871a7b20'
CPU_SHA = '6f3dc66c43d25b96d6bf5c7620231e115d8527e906ae00e9741bb3e55c5bd607'
SCHEMA = 'current_d50_hybrid_cp_qualification_v1'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--source-sha256')
    parser.add_argument('--model-sha256')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--mps-slot-clear', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    if not args.stage or any(not x or len(x) != 64 for x in
                            (args.source_sha256, args.model_sha256, args.protocol_sha256)):
        parser.error('Execution requires stage and reviewed source hashes')
    helper_path = CO / 'qualify.py'
    if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError('Frozen helper source changed')
    h = load_module(helper_path, 'hybrid_cp_frozen_helpers')
    folder = HERE / ('stage_' + args.stage + '_v1')
    folder.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(schema=SCHEMA, stage=args.stage, passed=False, completed=False,
                  sources_unchanged=False, alltruth_accessed=False, training_targets_accessed=False,
                  validation_targets_accessed=False, parameter_updates=0, no_fit=True,
                  native_candidate_qualified=False, score_measured=False,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  source_bindings={}, source_metadata={}, weight_bindings={}, dependencies={},
                  artifacts={}, artifact_bindings={}, resources={}, maximum_sampled_mps_driver_bytes=0)
    h.write(folder / 'started.json', result)
    progress = (folder / 'progress.jsonl').open('x')
    torch = None
    inside = False

    def record(**row):
        row = dict(elapsed_s=time.monotonic()-start, **row)
        text = json.dumps(row, allow_nan=False, sort_keys=True)
        progress.write(text+'\n'); progress.flush(); print(text, flush=True)

    def guard(_signal=None, _frame=None):
        nonlocal inside
        if inside:
            return
        inside = True
        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss = int(rss if sys.platform == 'darwin' else rss*1024)
            driver = int(torch.mps.driver_allocated_memory()) if torch is not None and args.stage.startswith('mps') else 0
            result['maximum_sampled_mps_driver_bytes'] = max(result['maximum_sampled_mps_driver_bytes'], driver)
            result['resources'] = dict(peak_rss_bytes=rss, sampled_driver_bytes=driver,
                                       elapsed_s=time.monotonic()-start)
            h.require(rss <= 12_000_000_000 and driver <= 16_000_000_000, 'Host/driver resource cap')
            h.require(time.monotonic()-start < 900, 'Fixed stage deadline')
        except BaseException:
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise
        finally:
            inside = False

    old_handler = signal.signal(signal.SIGALRM, guard)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    try:
        h.require(Path(sys.prefix).resolve() == Path('/Users/matt/anaconda3'), 'AP interpreter required')
        h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10 required')
        h.require(os.environ.get('PYTHONDONTWRITEBYTECODE') == '1', 'No bytecode writes')
        h.require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                  'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'BLAS1 required')
        h.require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'No MPS fallback')
        h.require(not any(x in sys.modules for x in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                  'Fresh numerical process required')
        h.require(not args.stage.startswith('mps') or args.mps_slot_clear, 'Sole MPS slot required')
        pins = {Path(__file__).resolve(): args.source_sha256, HERE/'model.py': args.model_sha256,
                HERE/'PROTOCOL.md': args.protocol_sha256, helper_path: HELPER_SHA}
        for path, digest in pins.items():
            result['source_metadata'][str(path)] = h.bind(path, digest)
        model = load_module(HERE/'model.py', 'hybrid_cp_model')
        for path, digest in model.authenticate().items():
            result['source_metadata'][str(path)] = h.bind(path, digest)
        result['source_bindings'] = {path: item['sha256'] for path,item in result['source_metadata'].items()}
        own = model.owner()
        h.require('numpy' not in sys.modules and 'torch' not in sys.modules, 'Source admission must be inert')
        cpu_path = CO/'stage_cpu_forward_v1/result.json'
        result['dependencies'][str(cpu_path)] = h.bind(cpu_path, CPU_SHA)
        cpu = json.loads(cpu_path.read_text())
        h.require(cpu['passed'] and cpu['completed'] and cpu['sources_unchanged'] and cpu['no_fit']
                  and not cpu['alltruth_accessed'] and cpu['parameter_updates'] == 0, 'Old CPU admission')
        for group in ('source_metadata','weight_bindings','artifact_bindings'):
            h.recheck(cpu[group]); result['dependencies'].update(cpu[group])
        for name in ('payloads','cotangents','probe_seal_sha256'):
            result[name] = cpu[name]
        result['probe_owner'] = str(cpu_path)
        result['probe_owner_sha256'] = CPU_SHA
        previous = {}
        for stage in STAGES[:STAGES.index(args.stage)]:
            rp = HERE/('stage_'+stage+'_v1')/'result.json'
            cp = rp.parent/'completion.json'
            rb, cb = h.bind(rp), h.bind(cp)
            rec, seal = json.loads(rp.read_text()), json.loads(cp.read_text())
            h.require(rec['schema'] == SCHEMA and rec['stage'] == stage and rec['passed']
                      and rec['completed'] and rec['sources_unchanged'] and seal['passed']
                      and seal['completed'] and seal['result_sha256'] == rb['sha256'], 'Prior stage not passed')
            h.require(rec['source_bindings'] == result['source_bindings'] and rec['probe_owner_sha256'] == CPU_SHA
                      and rec['no_fit'] and not rec['alltruth_accessed'] and rec['parameter_updates'] == 0,
                      'Prior lineage changed')
            h.require(rec['resources']['peak_rss_bytes'] <= 12_000_000_000
                      and rec['maximum_sampled_mps_driver_bytes'] <= 16_000_000_000
                      and rec['elapsed_s'] < 900 and rec['swap_growth_bytes'] <= 2*1024**3, 'Prior resource failure')
            for group in ('source_metadata','weight_bindings','artifact_bindings','dependencies'):
                h.recheck(rec[group]); result['dependencies'].update(rec[group])
            result['dependencies'].update({str(rp):rb,str(cp):cb})
            previous[stage] = rec
        paths,result['weight_bindings'] = h.weight_bindings()
        result['before_swap'] = h.swap_state()
        import numpy as np
        import torch as t
        torch = t
        h.require(np.__version__ == '1.26.4' and torch.__version__ == '2.7.0', 'Numerical versions changed')
        result['versions'] = dict(numpy=np.__version__,torch=torch.__version__)
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        h.require(not args.stage.startswith('mps') or torch.backends.mps.is_available(), 'MPS unavailable')
        enabled = args.stage.startswith('mps')

        def release():
            gc.collect()
            if enabled:
                torch.mps.synchronize(); torch.mps.empty_cache()
            guard()

        def output(pred):
            return h.prediction_array(np,own,pred)

        def compare(reference, value):
            expected = h.load_array(np,reference['prediction'])
            difference = float(np.max(np.abs(value.astype(np.float64)-expected)))
            return difference, value.tobytes() == expected.tobytes()

        if args.stage == 'mps_forward':
            result['outputs'],result['secants'],result['disabled_nulls'] = {},[],{}
            for rank in h.RANKS:
                weights = np.load(paths[rank],mmap_mode='r',allow_pickle=False)
                cot = h.load_array(np,cpu['cotangents'][str(rank)])
                values = h.load_bundle(np,cpu['payloads']['null'])
                P = h.torch_payload(torch,values,own.PKEYS,'cpu')
                card = model.card(rank,P=P,weights=weights,enabled=False)
                begin = time.monotonic()
                with torch.no_grad():
                    pred,_ = card.run(); value = output(pred)
                diagnostics = own.check_replay(card)
                reference = cpu['outputs'][str(rank)]['null']
                difference,exact = compare(reference,value)
                selectors = diagnostics['selection_cells'] == reference['diagnostics']['selection_cells']
                result['disabled_nulls'][str(rank)] = dict(prediction=h.save_array(np,folder/f'disabled_{rank:04d}.npy',value,result),
                    byte_exact=exact,ordered_selector_records_equal=selectors,diagnostics=diagnostics,
                    elapsed_s=time.monotonic()-begin)
                record(kind='disabled_null',rank=rank,byte_exact=exact,ordered_selectors_equal=selectors)
                h.require(exact and selectors and diagnostics['offloaded_products'] == 0,'Disabled inverse CPU parity')
                del card,P,values,pred,value
                release()
                result['outputs'][str(rank)] = {}
                losses = {}
                for case,amplitude in h.CASES:
                    values = h.load_bundle(np,cpu['payloads'][case])
                    P = h.torch_payload(torch,values,own.PKEYS,'cpu')
                    card = model.card(rank,P=P,weights=weights,enabled=True)
                    begin = time.monotonic()
                    with torch.no_grad():
                        pred,_ = card.run()
                        loss = float(h.scalar(torch,own,pred,cot,torch.float64))
                        value = output(pred)
                    diagnostics = own.check_replay(card)
                    reference = cpu['outputs'][str(rank)][case]
                    difference,exact = compare(reference,value)
                    item = dict(rank=rank,case=case,amplitude=amplitude,synthetic_loss=loss,
                        prediction=h.save_array(np,folder/f'row_{rank:04d}_{case}.npy',value,result),
                        diagnostics=diagnostics,maximum_absolute_difference=difference,cpu_byte_exact=exact,
                        selector_comparison=h.selector_difference(reference['diagnostics']['selection_cells'],diagnostics['selection_cells']),
                        elapsed_s=time.monotonic()-begin)
                    result['outputs'][str(rank)][case] = item
                    losses[case] = loss
                    record(kind='hybrid_forward',rank=rank,case=case,maximum_absolute_difference=difference,
                           card_elapsed_s=item['elapsed_s'],transferred_bytes=diagnostics['transferred_bytes'])
                    h.require(math.isfinite(loss) and difference <= 1e-5,'Hybrid all16 numerical gate')
                    h.require(diagnostics['offloaded_products'] == 30,'Exactly15 nonempty pushes and15 reads required')
                    del card,P,values,pred,value
                    release()
                for suffix,amplitude in (('025',.025),('050',.05)):
                    secant = (losses['plus'+suffix]-losses['minus'+suffix])/(2*amplitude)
                    old = next(x for x in cpu['secants'] if x['rank']==rank and x['amplitude']==amplitude)['symmetric_secant']
                    relative = abs(old-secant)/max(abs(old),abs(secant),1e-10)
                    item = dict(rank=rank,amplitude=amplitude,symmetric_secant=secant,cpu_secant=old,relative_error=relative)
                    result['secants'].append(item); record(kind='secant',**item)
                    h.require(math.isfinite(secant) and relative <= .05,'Unchanged5% scalar secant gate')
                del weights,cot
                release()
        else:
            result['gradients'],result['gradient_comparisons'] = {},{}
            for rank in h.RANKS:
                weights = np.load(paths[rank],mmap_mode='r',allow_pickle=False)
                values = h.load_bundle(np,cpu['payloads']['null'])
                cot = h.load_array(np,cpu['cotangents'][str(rank)])
                P = h.torch_payload(torch,values,own.PKEYS,'cpu',gradients=True)
                h.require(all(P[k].is_leaf and P[k].requires_grad for k in own.PKEYS),'Complete P leaves')
                card = model.card(rank,P=P,weights=weights,enabled=enabled)
                begin = time.monotonic()
                pred,_ = card.run(training=True)
                value = output(pred)
                diagnostics = card.diagnostics()
                own.check_domain(card)
                ref = previous['mps_forward']['outputs'][str(rank)]['null'] if enabled else cpu['outputs'][str(rank)]['null']
                difference,exact = compare(ref,value)
                item = dict(rank=rank,prediction=h.save_array(np,folder/f'vjp_primal_{rank:04d}.npy',value,result),
                            own_null_byte_exact=exact,forward_elapsed_s=time.monotonic()-begin,
                            forward_diagnostics=diagnostics)
                result['gradients'][str(rank)] = item
                record(kind='vjp_primal',rank=rank,own_null_byte_exact=exact)
                h.require(exact,'Checkpoint primal differs from own null')
                loss = h.scalar(torch,own,pred,cot,torch.float64)
                item['synthetic_loss'] = float(loss.detach())
                h.require(math.isfinite(item['synthetic_loss']),'Finite VJP scalar')
                loss.backward()
                after = own.check_replay(card)
                h.require(after['selection_cells']==diagnostics['selection_cells']
                          and after['variance_domain']==diagnostics['variance_domain'],'Checkpoint replay changed')
                gradients = {}
                for key in own.PKEYS:
                    h.require(P[key].grad is not None and bool(torch.isfinite(P[key].grad).all()),'Missing/nonfinite '+key)
                    gradients[key] = P[key].grad.detach().numpy().copy()
                    if key in own.INACTIVE_KEYS:
                        h.require(bool((gradients[key][0]==0).all()),'Inactive layer0 gradient')
                h.require(sum(x.size for x in gradients.values())==45200,'Complete P gradient count')
                h.require(all(not value.requires_grad and value.grad is None for key,value in P.items()
                              if key not in own.PKEYS),'Frozen fields acquired gradients')
                item.update(gradient=h.save_bundle(np,folder/f'gradient_{rank:04d}.npz',gradients,result),
                            diagnostics=after,forward_backward_elapsed_s=time.monotonic()-begin,
                            inactive_gradients_zero=True)
                record(kind='vjp_saved',rank=rank,card_elapsed_s=item['forward_backward_elapsed_s'])
                if enabled:
                    comparison = h.compare_gradients(np,previous['cpu_gradient']['gradients'][str(rank)],item,own)
                    result['gradient_comparisons'][str(rank)] = comparison
                    record(kind='full_group_vjp',rank=rank,**comparison)
                    h.require(comparison['passed'],'Unchanged full/group5% VJP gate')
                del loss,pred,value,card,P,values,gradients,weights,cot
                release()
        h.require('threadpoolctl' not in sys.modules and 'flopscope' not in sys.modules,'Forbidden AP module')
        for module in tuple(sys.modules.values()):
            path = getattr(module,'__file__',None)
            if path and Path(path).suffix=='.py' and Path(path).is_file():
                path = Path(path).resolve()
                if path.is_relative_to(HERE.parent):
                    h.require(str(path) in result['source_bindings'],'Unbound local source: '+str(path))
        for group in ('source_metadata','weight_bindings','artifact_bindings','dependencies'):
            h.recheck(result[group])
        result['after_swap'] = h.swap_state()
        result['swap_growth_bytes'] = result['after_swap']['used_bytes']-result['before_swap']['used_bytes']
        h.require(result['swap_growth_bytes'] <= 2*1024**3,'Swap growth cap')
        guard()
        result.update(passed=True,completed=True,sources_unchanged=True,
                      numerical_qualification_passed=args.stage=='mps_gradient')
    except BaseException as exc:
        result.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
        record(kind='failure',error=repr(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_handler)
        result['elapsed_s'] = time.monotonic()-start
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        progress.close()
        h.write(folder/'result.json',result)
        h.write(folder/'completion.json',dict(schema=SCHEMA,stage=args.stage,passed=result['passed'],
                  completed=result['completed'],result_sha256=h.sha(folder/'result.json')))
    print(json.dumps({key:result.get(key) for key in ('stage','passed','completed','elapsed_s','error')},sort_keys=True),flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
