"""One unmodified native d50 cold call with passive terminal feature export.

Inert without --run. No truth, fit, gradient, candidate edit, or extra FNP
capture arithmetic. The original paid/lifecycle spans are AST-qualified by
the frozen full-return capture's proof applied to THIS source file.
"""
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
from pathlib import Path
import resource
import sys
import time
import traceback
import types
import weakref

import common as c

BUDGET = 2**41


class Observer:
    """Immediately serialize small existing arrays; never retain frames/C/CP."""
    def __init__(self, module):
        self.predict_code = module._predict_binary.__code__
        self.gaussian_code = module.gaussian_raw.__code__
        self.select_code = module.select.__code__
        self.gnn_code = module.gnn_step.__code__
        self.predict_returns, self.gnn_returns, self.final_returns = 0, 0, 0
        self.fields, self.parameters = {}, {}
        self.variances, self.selectors = [], []
        self.raw = None

    @staticmethod
    def bytes_of(value):
        return dict(data=value.tobytes(),dtype=str(value.dtype),shape=tuple(value.shape),
                    strides=tuple(value.strides))

    def __call__(self, frame, event, answer):
        code = frame.f_code
        if code is self.gaussian_code and event == 'call':
            self.variances.append(self.bytes_of(frame.f_locals['v']))
        elif code is self.select_code and event == 'return':
            c.require(isinstance(answer,tuple) and len(answer) == 2, 'Original selector result')
            self.selectors.append((self.bytes_of(answer[0]),self.bytes_of(answer[1]),
                                   float(frame.f_locals['floor'])))
        elif code is self.gnn_code and event == 'return':
            self.gnn_returns += 1
            if frame.f_locals['l'] == 15:
                self.final_returns += 1
                c.require(self.final_returns == 1 and isinstance(answer,tuple) and len(answer) == 2,
                          'One actual final GNN return')
                self.fields.update(hidden=self.bytes_of(answer[1]),delta=self.bytes_of(answer[0]),
                                   sigma=self.bytes_of(frame.f_locals['sigma']))
                # Whole existing arrays: no participant indexing/cast/recomputed factors.
                self.parameters = {key:self.bytes_of(frame.f_locals['P'][key])
                                   for key in ('head','head_b','gain','rms')}
        elif code is self.predict_code and event == 'return':
            self.predict_returns += 1
            c.require(self.predict_returns == 1 and frame.f_locals['l'] == 15, 'One complete original card')
            means = frame.f_locals['means']
            c.require(len(means) == 16 and all(str(v.dtype) == 'float64' and tuple(v.shape) == (1024,)
                                             for v in means), 'Existing sixteen raw mean vectors')
            self.raw = b''.join(value.tobytes() for value in means)
            self.fields.update(host=self.bytes_of(frame.f_locals['host']),
                               hook_delta=self.bytes_of(frame.f_locals['dm']))

    def export_bytes(self):
        c.require(self.predict_returns == 1 and self.gnn_returns == 16 and self.final_returns == 1
                  and len(self.variances) == 16 and len(self.selectors) == 47 and self.raw is not None,
                  'Complete passive capture roster')
        packet = dict(raw=self.raw,fields=dict(self.fields),parameters=dict(self.parameters),
                      variances=tuple(self.variances),selectors=tuple(self.selectors))
        self.clear()
        return packet

    def clear(self):
        self.raw = None
        self.fields.clear()
        self.parameters.clear()
        self.variances.clear()
        self.selectors.clear()


def decode(np, item):
    return np.frombuffer(item['data'],dtype=item['dtype']).reshape(item['shape'])


def diagnostics(np, packet):
    variances, selectors = {}, {}
    for layer,item in enumerate(packet['variances']):
        value = decode(np,item)
        c.require(value.shape == (1024,) and value.dtype == np.float64, 'Actual prevariance type')
        finite = bool(np.isfinite(value).all())
        variances[str(layer)] = dict(finite=finite,minimum=float(value.min()) if finite else None,
            below_floor_count=int((value < 1e-10).sum()),raw_sha256=hashlib.sha256(item['data']).hexdigest())
    keys = [f'{l}:{slot}' for l in range(16) for slot in range(2 if l == 15 else 3)]
    for key,(left,right,floor) in zip(keys,packet['selectors']):
        ids,mask = decode(np,left),decode(np,right)
        c.require(ids.ndim == 1 and str(ids.dtype) in ('int32','int64') and mask.shape == (1024,)
                  and mask.dtype == np.bool_ and floor == .01, 'Actual selector dtype/floor')
        order,membership = ids.tolist(),np.flatnonzero(mask).tolist()
        c.require(len(order) % 32 == 0 and len(set(order)) == len(order) and sorted(order) == membership
                  and all(0 <= x < 1024 for x in order), 'Selector ordered support integrity')
        selectors[key] = dict(ordered_ids=order,membership=membership,count=len(order),floor=floor,
            indices_sha256=hashlib.sha256(left['data']).hexdigest(),mask_sha256=hashlib.sha256(right['data']).hexdigest())
    c.require(len(variances) == 16 and set(selectors) == set(keys), 'All16 prevariances/all47 selectors')
    c.require(all(x['finite'] and x['minimum'] >= 1e-10 and x['below_floor_count'] == 0
                  for x in variances.values()), 'No nonfinite/floored prevariance')
    c.require(all(selectors[f'0:{slot}']['ordered_ids'] == list(range(1024)) for slot in range(3)),
              'Unchanged Gaussian layer0 supports')
    return dict(variance_domain=variances,selection_cells=selectors,variance_count=16,selector_count=47,
        added_participant_array_operations=0,retention='Small byte packets only; no live moment views/frames')


def replay(np, stage, module, packet, arrays):
    """Same AST/grouping, same NumPy2 weak scalar promotion, AFTER saved/reset bill."""
    with np.load(c.PAYLOAD,allow_pickle=False) as data:
        payload = {key:data[key].copy() for key in data.files}
    c.require(set(payload) == set(module._GNN_LITERAL) == set(module._GNN_PARAMS), 'Exact original P keys')
    checks = {}
    for key,value in payload.items():
        literal = np.asarray(module._GNN_LITERAL[key],dtype=np.float32)
        checks[key] = (value.dtype == np.float32 and np.isfinite(value).all()
            and value.shape == literal.shape and value.tobytes() == literal.tobytes()
            and value.tobytes() == module._GNN_PARAMS[key].tobytes())
    c.require(all(checks.values()), 'Original payload/literal/actual native P array bytes')
    for key,item in packet['parameters'].items():
        captured = decode(np,item)
        c.require(captured.dtype == np.float32 and captured.shape == payload[key].shape
                  and captured.tobytes() == payload[key].tobytes(), 'Captured final P bytes: '+key)
    tree = ast.parse(c.CANDIDATE.read_text())
    functions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == 'gnn_step']
    c.require(len(functions) == 1, 'Unique frozen active GNN source')
    expressions = [node.value for node in functions[0].body if isinstance(node,ast.Assign)
                   and any(isinstance(target,ast.Name) and target.id == 'delta' for target in node.targets)]
    c.require(len(expressions) == 1, 'Exact original final delta expression')
    expr = ast.Expression(expressions[0])
    program = compile(ast.fix_missing_locations(expr),str(c.CANDIDATE),'eval')
    # Do not widen P scalars or regroup multiplications: original expression runs verbatim.
    delta = eval(program,{'fnp':np,'__builtins__':{}},
                 dict(l=15,state=arrays['hidden'].copy(),sigma=arrays['sigma'],P=payload))
    widened = decode(np,packet['fields']['hook_delta'])
    raw_final = arrays['host'] + delta.astype(np.float64)
    served_final = np.maximum(raw_final,0.0).astype(np.float32)
    result = dict(original_payload_arrays_byte_exact=True,original_P_unchanged=True,
        captured_parameter_bytes_exact=True,expression_ast_sha256=hashlib.sha256(ast.dump(expr).encode()).hexdigest(),
        expression=ast.unparse(expressions[0]),
        scalar_dtypes={key:str(payload[key][15].dtype) for key in ('rms','gain','head_b')},
        native_delta_dtype=str(arrays['delta'].dtype),replay_delta_dtype=str(delta.dtype),
        delta_byte_exact=delta.dtype == np.float32 and delta.tobytes() == arrays['delta'].tobytes(),
        widened_hook_delta_byte_exact=widened.dtype == np.float64 and widened.tobytes() == arrays['delta'].astype(np.float64).tobytes(),
        final_raw_byte_exact=raw_final.tobytes() == arrays['raw'][15].tobytes(),
        final_served_byte_exact=served_final.tobytes() == arrays['served'][15].tobytes(),
        offline_only=True,native_bill_already_saved_and_context_reset=True)
    stage.result['head_replay'] = result
    c.require(all(result[key] for key in ('delta_byte_exact','widened_hook_delta_byte_exact',
        'final_raw_byte_exact','final_served_byte_exact')), 'Original native terminal block replay must be BYTE exact')


def main():
    args = c.arguments()
    if not args.run:
        return 0
    stage = c.Stage(args,Path(__file__))
    result,rank,legacy = stage.result,args.rank,stage.legacy
    observer = None
    try:
        # The frozen proof reads this file, not the old FQ implementation.
        proof_globals = dict(stage.template.__dict__,__file__=str(Path(__file__).resolve()))
        proof = types.FunctionType(stage.template.ownership_proof.__code__,proof_globals)
        result['ownership_proof'] = proof()
        result['ownership_proof']['proof_source_sha256'] = c.TEMPLATE_SHA
        versions = {name:importlib.metadata.version(name) for name in ('numpy','flopscope','whestbench','threadpoolctl')}
        c.require(versions == dict(numpy='2.4.6',flopscope='0.12.0',whestbench='0.16.0',threadpoolctl='3.6.0'),
                  'Fixed official native library versions')
        c.require(not any(x in sys.modules for x in ('numpy','torch','flopscope','threadpoolctl')),
                  'All source/payload/weight pins precede numerical imports')
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits,threadpool_info
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        module = c.module(c.CANDIDATE,'terminal_linear_unmodified_d50')
        c.require(all(fn.__code__.co_filename == str(c.CANDIDATE)
                      for fn in (module._predict_binary,module.gnn_step,module.gaussian_raw,module.select)),
                  'Observe exact active candidate functions')
        observer = Observer(module)
        result.update(environment=dict(versions=versions,device='native CPU'),cold=True,all_primary_cold=True,
            includes_initialization=True,initialization_subtracted=False,candidate_source_edited=False,
            input_preparation_in_participant_bill=False,own_complete_bill=True,
            reset_scope='Harness only, after closed native context bill; no candidate reset')

        def save_receipt(path,record):
            legacy.write(path,record)
            item = stage.pin(path)
            result['artifacts'][str(path)] = item['sha256']
            return item

        with threadpool_limits(limits=1):
            result['threadpools'] = threadpool_info()
            c.require(all(p['num_threads'] == 1 for p in result['threadpools'] if p.get('user_api') == 'blas'),
                      'Every reported BLAS controller is one-threaded')
            weights = np.load(stage.weights_path,mmap_mode='r',allow_pickle=False)
            result['weights_read'] = True
            c.require(weights.shape == (16,1024,1024) and weights.dtype == np.float32
                      and np.isfinite(weights).all(), 'Canonical finite official weights')
            input_started = time.monotonic()
            with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as input_ctx:
                weight_layers=[fnp.array(w,dtype=fnp.float32) for w in weights]
            input_record = dict(call=0,rank=rank,**legacy.metrics(input_ctx,input_started),
                charged_to_participant=False,layers=16,context_closed=True,
                weight_file_sha256=result['weights']['sha256'])
            input_record['official_fnp_type'] = all(isinstance(w,fnp.ndarray) and w.dtype == np.float32
                and w.shape == (1024,1024) for w in weight_layers)
            input_record['exact_bytes'] = all(w.tobytes() == weights[l].tobytes() for l,w in enumerate(weight_layers))
            input_record['independent_weight_buffers'] = all(not np.may_share_memory(np.asarray(w),weights) for w in weight_layers)
            input_record['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            result['input_preparation'] = legacy.json_safe(input_record)
            input_bill = stage.folder/'input_bill.json'
            legacy.write(input_bill,input_record)
            input_ref = weakref.ref(input_ctx)
            input_reset = legacy.reset_saved_history(flops,input_ctx,input_bill)
            del input_ctx
            gc.collect()
            input_reset.update(call=0,rank=rank,context_released=input_ref() is None,complete=True,
                               finished_utc=datetime.now(timezone.utc).isoformat())
            result['input_lifecycle'] = input_reset
            result['input_bill'] = stage.pin(input_bill)
            result['artifacts'][str(input_bill)] = result['input_bill']['sha256']
            result['input_lifecycle_receipt'] = save_receipt(stage.folder/'input_lifecycle.json',input_reset)
            legacy.guard(input_record)
            c.require(input_record['official_fnp_type'] and input_record['exact_bytes']
                      and input_record['independent_weight_buffers'] and input_record['flops'] == 16*1024*1024
                      and input_reset['closed_context_bill_unchanged_by_reset'] and input_reset['context_released'],
                      'Unchanged official input ownership and post-bill release')
            stage.guard()
            mlp = MLP(width=1024,depth=16,weights=weight_layers,
                      seed=int(derive_estimator_seed(stage.mlp_seed)),name=f'birth-dir-{rank}')
            del weights,weight_layers
            estimator,ctx,packet,prediction_error = None,None,None,None
            c.require(sys.getprofile() is None, 'Never replace an existing profiler')
            result['network_calls'] = 1
            started = time.monotonic()
            sys.setprofile(observer)
            try:
                with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as ctx:
                    if estimator is None: estimator=module.Estimator()
                    served=estimator.predict(mlp,BUDGET)
                    assert isinstance(served,np.ndarray) and served.dtype==np.float32 and served.shape==(16,1024)
                    materialized=materialise_predictions(served,depth=16,width=1024)
                    packet = observer.export_bytes()
            except BaseException:
                prediction_error = traceback.format_exc()
            finally:
                sys.setprofile(None)
                observer.clear()
            c.require(ctx is not None, 'Native budget entered')
            trial = dict(call=0,rank=rank,**legacy.metrics(ctx,started),cold=True,includes_initialization=True,
                multiplier=max(.1,float(ctx.flops_used)/BUDGET),observed_predict_calls=observer.predict_returns,
                capture_serialization_inside_context=True)
            if prediction_error is not None:
                trial['prediction_error'] = prediction_error
            result['complete_bill'] = legacy.json_safe(trial)
            bill = stage.folder/'bill.json'
            legacy.write(bill,trial)
            context_ref = weakref.ref(ctx)
            lifecycle = legacy.reset_saved_history(flops,ctx,bill)
            del ctx
            gc.collect()
            lifecycle.update(call=0,rank=rank,context_released=context_ref() is None,complete=True,
                             finished_utc=datetime.now(timezone.utc).isoformat())
            result['lifecycle'] = lifecycle
            result['bill'] = stage.pin(bill)
            result['artifacts'][str(bill)] = result['bill']['sha256']
            result['lifecycle_receipt'] = save_receipt(stage.folder/'lifecycle.json',lifecycle)
            c.require(prediction_error is None, 'Original native call failed; attempted bill preserved')
            legacy.guard(trial)
            c.require(lifecycle['closed_context_bill_unchanged_by_reset'] and lifecycle['context_released'],
                      'Original closed-context metrics survive reset; context released')
            stage.guard()
            c.require(packet is not None and len(packet['raw']) == 16*1024*8
                      and served.tobytes() == materialized.tobytes(), 'Original materialization/raw export')
            arrays = {key:decode(np,packet['fields'][key]) for key in ('host','hidden','sigma','delta')}
            arrays.update(raw=np.frombuffer(packet['raw'],dtype=np.float64).reshape(16,1024),
                          served=np.frombuffer(materialized.tobytes(),dtype=np.float32).reshape(16,1024))
            expected = dict(host=((1024,),'float64'),hidden=((1024,64),'float32'),
                            sigma=((1024,),'float64'),delta=((1024,),'float32'),
                            raw=((16,1024),'float64'),served=((16,1024),'float32'))
            for key,value in arrays.items():
                c.require((tuple(value.shape),str(value.dtype)) == expected[key] and np.isfinite(value).all(),
                          'Actual capture shape/dtype/finite: '+key)
                result['captures'][key] = stage.save(np,key+'.npy',value)
            result['observed_layouts'] = {key:{k:v for k,v in item.items() if k != 'data'}
                for key,item in packet['fields'].items()}
            result['served_reconstruction_byte_exact'] = (
                np.maximum(arrays['raw'],0.0).astype(np.float32).tobytes() == arrays['served'].tobytes())
            c.require((arrays['served'] >= 0).all() and result['served_reconstruction_byte_exact'],
                      'All16 native clamp/return reconstruction')
            result['diagnostics'] = diagnostics(np,packet)
            result['all16_domains_present'],result['all47_selectors_present'] = True,True
            result['observed_predict_calls'],result['observed_gnn_calls'] = observer.predict_returns,observer.gnn_returns
            replay(np,stage,module,packet,arrays)
            result['archived_cold_reference'] = dict(required=stage.reference is not None)
            if stage.reference is not None:
                old = stage.reference
                served_ref,raw_ref = stage.load(np,old['served']),stage.load(np,old['raw_unclamped'])
                checks = dict(required=True,served_byte_exact=arrays['served'].tobytes() == served_ref.tobytes(),
                    raw_byte_exact=arrays['raw'].tobytes() == raw_ref.tobytes(),
                    complete_cold_flops_exact=trial['flops'] == old['complete_bill']['flops'],
                    operation_count_exact=trial['operation_count'] == old['complete_bill']['operation_count'],
                    ordered_selectors_exact=result['diagnostics']['selection_cells'] == old['diagnostics']['selection_cells'])
                result['archived_cold_reference'] = checks
                c.require(all(checks.values()), 'Initial35/100 original cold FQ native identity')
                del served_ref,raw_ref
            c.require('torch' not in sys.modules and observer.raw is None and sys.getprofile() is None,
                      'No Torch/live capture references or profiler')
            del mlp,estimator,served,materialized,arrays,packet
            observer = None
            gc.collect()
        stage.guard()
        stage.finish()
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
    finally:
        if observer is not None:
            if sys.getprofile() is observer:
                sys.setprofile(None)
            observer.clear()
        stage.close()
    return 0 if result.get('passed') and result.get('completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
