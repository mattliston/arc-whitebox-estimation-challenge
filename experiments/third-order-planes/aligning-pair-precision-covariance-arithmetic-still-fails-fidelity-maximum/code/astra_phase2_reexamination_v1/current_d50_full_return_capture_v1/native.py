"""One cold native forward, passively exporting its existing FP64 mean rows.

No candidate edits, surrogate arithmetic, targets, gradients, or fitting.
The official input/predict/materialise blocks and public post-bill reset
semantics are source-compared to frozen BD/native_v2.py before imports.
"""
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import math
from pathlib import Path
import resource
import sys
import time
import traceback
import weakref

import common as c

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'birth_direction_fit_v1/native_v2.py'
ORIGINAL_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
BUDGET = 2**41


def require(value, message):
    if not value:
        raise RuntimeError(message)


def ownership_proof():
    """AST identity of paid blocks and post-context bill/reset operations.

Only the Python byte-export statement follows the original predict block.
The reset helper is imported intact, and the closed-context references are
released in the same four-statement order. This does not claim whole-harness
identity: singleton admission, observations and output schemas are new.
"""
    require(c.sha(ORIGINAL) == ORIGINAL_SHA, 'Frozen native meter changed')
    old, new = ast.parse(ORIGINAL.read_text()), ast.parse(Path(__file__).read_text())

    class Names(ast.NodeTransformer):
        def visit_Attribute(self, node):
            node = self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == 'legacy':
                return ast.Name(id=node.attr, ctx=node.ctx)
            return node

    def dump(node):
        return ast.dump(Names().visit(ast.fix_missing_locations(node)), include_attributes=False)

    def context(tree, name):
        found = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                 and len(node.items) == 1 and isinstance(node.items[0].optional_vars, ast.Name)
                 and node.items[0].optional_vars.id == name]
        require(len(found) == 1, 'Unique paid context: ' + name)
        return found[0]

    proofs = {}
    for name in ('input_ctx', 'ctx'):
        a, b = context(old, name), context(new, name)
        require(dump(a.items[0]) == dump(b.items[0]), 'Unchanged context boundary')
        require([dump(x) for x in a.body] == [dump(x) for x in b.body[:len(a.body)]],
                'Original paid arithmetic prefix: ' + name)
        if name == 'input_ctx':
            require(len(a.body) == len(b.body), 'No input arithmetic added')
        else:
            require(len(b.body) == len(a.body) + 1
                    and dump(b.body[-1]) == dump(ast.parse('packet = observer.export_bytes()').body[0]),
                    'Only byte export appended inside prediction context')
        proofs[name] = hashlib.sha256(dump(a).encode()).hexdigest()

    def sequence(tree, target, length):
        found = []
        for node in ast.walk(tree):
            for field in ('body', 'orelse', 'finalbody'):
                body = getattr(node, field, None)
                if not isinstance(body, list):
                    continue
                for i, item in enumerate(body):
                    if isinstance(item, ast.Assign) and len(item.targets) == 1 \
                            and isinstance(item.targets[0], ast.Name) and item.targets[0].id == target:
                        found.append(body[i:i+length])
        require(len(found) == 1 and len(found[0]) == length, 'Unique lifecycle span: ' + target)
        return [dump(x) for x in found[0]]

    for name in ('input_ref', 'context_ref'):
        a, b = sequence(old, name, 4), sequence(new, name, 4)
        require(a == b, 'Closed bill/reset/release order: ' + name)
        proofs[name] = hashlib.sha256('\n'.join(a).encode()).hexdigest()
    for arguments in ('input_bill, input_record', 'bill, trial'):
        expected = dump(ast.parse('write(' + arguments + ')').body[0])
        require(any(isinstance(x, ast.Expr) and dump(x) == expected for x in ast.walk(old))
                and any(isinstance(x, ast.Expr) and dump(x) == expected for x in ast.walk(new)),
                'Original saved bill write is present')
    return dict(original_native_sha256=ORIGINAL_SHA, paid_blocks_ast_identical=True,
                original_numeric_prefix_unchanged=True, reset_order_ast_identical=True,
                extra_prediction_work='Existing-array byte serialization only', block_sha256=proofs)


class Observer:
    """No frame/local dictionaries, W/C/CP state, or recomputed moments retained."""
    def __init__(self, module):
        self.predict_code = module._predict_binary.__code__
        self.gaussian_code = module.gaussian_raw.__code__
        self.select_code = module.select.__code__
        self.means = None
        self.variances, self.selectors = [], []
        self.predict_returns = 0

    @staticmethod
    def bytes_of(value):
        # ndarray.tobytes is a diagnostic export, not array arithmetic.
        return dict(data=value.tobytes(), dtype=str(value.dtype), shape=tuple(value.shape))

    def __call__(self, frame, event, answer):
        code = frame.f_code
        if code is self.gaussian_code and event == 'call':
            # gaussian_raw immediately rebinds v=max(v,1e-10). Entry is required.
            self.variances.append(self.bytes_of(frame.f_locals['v']))
        elif code is self.select_code and event == 'return':
            require(isinstance(answer, tuple) and len(answer) == 2, 'Original selector return')
            self.selectors.append((self.bytes_of(answer[0]), self.bytes_of(answer[1]),
                                   float(frame.f_locals['floor'])))
        elif code is self.predict_code and event == 'return':
            self.predict_returns += 1
            require(self.means is None, 'Exactly one active predict return')
            self.means = tuple(frame.f_locals['means'])

    def export_bytes(self):
        require(self.predict_returns == 1 and self.means is not None and len(self.means) == 16,
                'One full original return with sixteen existing means')
        require(all(tuple(x.shape) == (1024,) and str(x.dtype) == 'float64' for x in self.means),
                'Existing native means must be FP64 vectors')
        require(len(self.variances) == 16 and len(self.selectors) == 47, 'Full observer roster')
        packet = dict(raw=b''.join(x.tobytes() for x in self.means),
                      variances=tuple(self.variances), selectors=tuple(self.selectors))
        self.clear()
        return packet

    def clear(self):
        self.means = None
        self.variances.clear()
        self.selectors.clear()


def diagnostics(np, packet, reference_cpu):
    def array(item):
        return np.frombuffer(item['data'], dtype=item['dtype']).reshape(item['shape'])

    variances, selectors = {}, {}
    for layer, item in enumerate(packet['variances']):
        value = array(item)
        require(value.shape == (1024,) and value.dtype == np.float64, 'Native variance frame')
        finite = bool(np.isfinite(value).all())
        variances[str(layer)] = dict(dtype=str(value.dtype), finite=finite,
            minimum=float(value.min()) if finite else None,
            below_floor_count=int((value < 1e-10).sum()),
            raw_sha256=hashlib.sha256(item['data']).hexdigest())
    keys = [f'{layer}:{slot}' for layer in range(16) for slot in range(2 if layer == 15 else 3)]
    for key, (left, right, floor) in zip(keys, packet['selectors']):
        ids, mask = array(left), array(right)
        require(ids.ndim == 1 and str(ids.dtype) in ('int32', 'int64')
                and mask.shape == (1024,) and mask.dtype == np.bool_ and floor == .01,
                'Original selector types/floor')
        order, membership = ids.tolist(), np.flatnonzero(mask).tolist()
        require(len(order) % 32 == 0 and len(set(order)) == len(order)
                and sorted(order) == membership and all(0 <= x < 1024 for x in order),
                'Native selector roster/membership')
        selectors[key] = dict(ordered_ids=order, membership=membership, count=len(order), floor=floor,
            indices_sha256=hashlib.sha256(left['data']).hexdigest(),
            mask_sha256=hashlib.sha256(right['data']).hexdigest())
    require(set(variances) == {str(l) for l in range(16)} and set(selectors) == set(keys),
            'Sixteen variance / forty-seven selector records')
    require(all(x['finite'] and x['minimum'] >= 1e-10 and x['below_floor_count'] == 0
                for x in variances.values()), 'Positive unfloored native prevariances')
    require(all(selectors[f'0:{s}']['ordered_ids'] == list(range(1024)) for s in range(3)),
            'Full Gaussian first-layer selector')
    old = reference_cpu['diagnostics']['selection_cells']
    return dict(variance_domain=variances, selection_cells=selectors,
        variance_count=16, selector_count=47,
        old_cpu_selector_memberships_equal=all(selectors[k]['membership'] == old[k]['membership'] for k in keys),
        old_cpu_selector_orders_equal=all(selectors[k]['ordered_ids'] == old[k]['ordered_ids'] for k in keys),
        variance_bytes_compared_to_cpu=False, old_native_selector_receipt_available=False,
        added_participant_array_operations=0, gaussian_observation='Unfloored v at call entry',
        retention='Only existing mean vectors; other small observations serialized immediately')


def main():
    args = c.arguments('native')
    if not args.run:
        return 0
    stage = c.Stage('native', args, Path(__file__))
    result, rank = stage.result, args.rank
    observer = None
    try:
        result['ownership_proof'] = ownership_proof()
        stage.pin(ORIGINAL, ORIGINAL_SHA)
        legacy = c.module(ORIGINAL, 'full_return_original_native_helpers')
        for path, digest in legacy.PINS.items():
            stage.pin(path, digest)
        versions = {name: importlib.metadata.version(name)
                    for name in ('numpy', 'flopscope', 'whestbench', 'threadpoolctl')}
        require(versions == dict(numpy='2.4.6', flopscope='0.12.0', whestbench='0.16.0', threadpoolctl='3.6.0'),
                'Original official native library frame')
        require(not any(x in sys.modules for x in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'Complete source/payload/weight authentication before numerical imports')
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits, threadpool_info
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        module = c.module(stage.candidate, 'full_return_original_candidate')
        require(module._predict_binary.__code__.co_filename == str(stage.candidate),
                'Observe the actual active candidate function')
        observer = Observer(module)
        result.update(environment=dict(versions=versions, device='native CPU'),
            network_calls=0, gradient_calls=0, parameter_updates=0,
            cold=True, all_primary_cold=True, includes_initialization=True,
            input_preparation_in_participant_bill=False, initialization_subtracted=False,
            original_reference_includes_initialization=stage.reference_native['includes_initialization'],
            original_reference_flops=stage.reference_native['flops'],
            original_reference_is_comparable_cold_bill=bool(stage.reference_native['includes_initialization']),
            new_derivative_qualified=False, training_admitted=False,
            raw_capture_is_existing_return=True, candidate_source_edited=False,
            reset_scope='Only after closed context bill is saved; no candidate reset')

        def save_receipt(path, record):
            legacy.write(path, record)
            item = stage.pin(path)
            result['artifacts'][str(path)] = item['sha256']
            return item

        with threadpool_limits(limits=1):
            result['threadpools'] = threadpool_info()
            require(all(p['num_threads'] == 1 for p in result['threadpools'] if p.get('user_api') == 'blas'),
                    'Every reported BLAS controller must be one-threaded')
            weights = np.load(stage.weights_path, mmap_mode='r', allow_pickle=False)
            require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32
                    and np.isfinite(weights).all(), 'Authenticated finite official weights')
            input_started = time.monotonic()
            with flops.BudgetContext(flop_budget=BUDGET,wall_time_limit_s=120,quiet=True) as input_ctx:
                weight_layers=[fnp.array(w,dtype=fnp.float32) for w in weights]
            input_record = dict(call=0, rank=rank, **legacy.metrics(input_ctx, input_started),
                charged_to_participant=False, layers=16, context_closed=True,
                weight_file_sha256=stage.bindings[str(stage.weights_path)]['sha256'])
            input_record['official_fnp_type'] = all(isinstance(w, fnp.ndarray) and w.dtype == np.float32
                and w.shape == (1024, 1024) for w in weight_layers)
            input_record['exact_bytes'] = all(w.tobytes() == weights[l].tobytes()
                                               for l, w in enumerate(weight_layers))
            input_record['independent_weight_buffers'] = all(
                not np.may_share_memory(np.asarray(w), weights) for w in weight_layers)
            input_record['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            result['input_preparation'] = legacy.json_safe(input_record)
            input_bill = stage.folder / 'input_bill.json'
            legacy.write(input_bill, input_record)
            input_ref = weakref.ref(input_ctx)
            input_reset = legacy.reset_saved_history(flops, input_ctx, input_bill)
            del input_ctx
            gc.collect()
            input_reset.update(call=0, rank=rank, context_released=input_ref() is None, complete=True,
                               finished_utc=datetime.now(timezone.utc).isoformat())
            result['input_lifecycle'] = input_reset
            result['input_bill'] = stage.pin(input_bill)
            result['artifacts'][str(input_bill)] = result['input_bill']['sha256']
            result['input_lifecycle_receipt'] = save_receipt(stage.folder / 'input_lifecycle.json', input_reset)
            legacy.guard(input_record)
            require(input_record['official_fnp_type'] and input_record['exact_bytes']
                    and input_record['independent_weight_buffers'] and input_record['flops'] == 16*1024*1024,
                    'Original official-input ownership/bill')
            require(input_reset['closed_context_bill_unchanged_by_reset'] and input_reset['context_released'],
                    'Input budget released after immutable bill')
            stage.guard()
            mlp = MLP(width=1024, depth=16, weights=weight_layers,
                      seed=int(derive_estimator_seed(stage.mlp_seed)), name=f'birth-dir-{rank}')
            del weights, weight_layers
            estimator = None
            ctx = None
            packet = None
            prediction_error = None
            require(sys.getprofile() is None, 'No existing profiler may be replaced')
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
            require(ctx is not None, 'Prediction context was entered')
            trial = dict(call=0, rank=rank, **legacy.metrics(ctx, started),
                         includes_initialization=True, cold=True,
                         multiplier=max(.1, float(ctx.flops_used)/BUDGET),
                         capture_serialization_inside_context=True,
                         observed_predict_calls=observer.predict_returns)
            if prediction_error is not None:
                trial['prediction_error'] = prediction_error
            result['complete_bill'] = legacy.json_safe(trial)
            bill = stage.folder / 'bill.json'
            legacy.write(bill, trial)
            context_ref = weakref.ref(ctx)
            lifecycle = legacy.reset_saved_history(flops, ctx, bill)
            del ctx
            gc.collect()
            lifecycle.update(call=0, rank=rank, context_released=context_ref() is None, complete=True,
                             finished_utc=datetime.now(timezone.utc).isoformat())
            result['lifecycle'] = lifecycle
            result['bill'] = stage.pin(bill)
            result['artifacts'][str(bill)] = result['bill']['sha256']
            result['lifecycle_receipt'] = save_receipt(stage.folder / 'lifecycle.json', lifecycle)
            require(prediction_error is None, 'Original prediction failed; complete attempted bill preserved')
            legacy.guard(trial)
            require(lifecycle['closed_context_bill_unchanged_by_reset'] and lifecycle['context_released'],
                    'Prediction budget released after immutable complete bill')
            stage.guard()
            require(packet is not None and len(packet['raw']) == 16*1024*8,
                    'Only existing all16 FP64 means exported')
            require(served.tobytes() == materialized.tobytes(), 'Original materialization bytes')
            values = np.frombuffer(materialized.tobytes(), dtype=np.float32).reshape(16, 1024)
            raw = np.frombuffer(packet['raw'], dtype=np.float64).reshape(16, 1024)
            clipped = np.maximum(raw, 0.)
            reference = stage.load(np, dict(path=stage.reference_native['output'],
                sha256=stage.reference_native['output_sha256'], dtype='float32', shape=[16, 1024],
                raw_sha256=stage.reference_native['output_bytes_sha256']))
            result['raw_unclamped'] = stage.save(np, 'raw_unclamped.npy', raw)
            result['pre_return'] = stage.save(np, 'pre_return.npy', clipped)
            result['served'] = stage.save(np, 'served.npy', values)
            result['prediction'] = result['served']
            result['served_reconstruction_byte_exact'] = clipped.astype(np.float32).tobytes() == values.tobytes()
            result['old_native_served_byte_exact'] = values.tobytes() == reference.tobytes()
            result['clipping_count'] = int((raw < 0).sum())
            result['clipping_count_by_layer'] = [int((row < 0).sum()) for row in raw]
            result['observed_predict_calls'] = observer.predict_returns
            result['diagnostics'] = diagnostics(np, packet, stage.reference_cpu)
            result['all16_domains_present'] = True
            result['all47_selectors_present'] = True
            result['raw_capture_numeric_bytes'] = len(packet['raw'])
            result['cold_bill_matches_original_if_cold'] = (
                trial['flops'] == stage.reference_native['flops'] if stage.reference_native['includes_initialization'] else None)
            require(np.isfinite(raw).all() and np.isfinite(values).all() and (values >= 0).all()
                    and result['served_reconstruction_byte_exact'] and result['old_native_served_byte_exact'],
                    'Exact original native endpoint and unclamped reconstruction')
            if stage.reference_native['includes_initialization']:
                require(result['cold_bill_matches_original_if_cold'], 'Exact original cold35 complete bill')
            require('torch' not in sys.modules and observer.means is None and sys.getprofile() is None,
                    'No Torch, retained frame/state or profiler after capture')
            del mlp, estimator, served, materialized, values, raw, clipped, reference, packet, observer
            observer = None
            gc.collect()
        stage.guard()
        stage.finish()
    except BaseException:
        result.update(passed=False, completed=False, hashes_unchanged=False, sources_unchanged=False,
                      error=traceback.format_exc())
    finally:
        if observer is not None:
            if sys.getprofile() is observer:
                sys.setprofile(None)
            observer.clear()
        stage.close()
    return 0 if result.get('passed') and result.get('completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
