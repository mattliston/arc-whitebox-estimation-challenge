"""Inert, paid, weights-only native d50 initial-Gram witness.

Only --run admits numerical imports.  This is NOT an Estimator, a complete
initialization bill, a CPU-seed experiment, a qualification retry, or a score.
Exact last source definitions are compiled without importing the participant
or evaluating any P/cstar literal.  The public diagnostic reset occurs only
after each closed context's bill is saved; matrix state is never reset.
"""
import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
NEW, P2 = HERE.parent, HERE.parent.parent
BD = NEW / 'birth_direction_fit_v1'
BR = NEW / 'current_d50_native_p_bridge_v1'
OFFPY = P2 / 'data_gate/.venv'
SITE = OFFPY / 'lib/python3.11/site-packages'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
PARENT = BD / 'build_terminal_v1/estimator.py'
PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
NATIVE_SHA = '8a1b888be28a7ba6d7f8e020bb9b1f2367786e189d952364710b34517ca2b35b'
VERIFIER_SHA = '5528601148c98384de72af4ea7f7f39eed1d4657c51617910cd2be466299a608'
REFERENCE_SHA = '039c98a5acffa4e88440ef331b7efafd78b4c56163043dcbc670c0eedc1cd333'
WATCH_REFERENCE_SHA = '369b557223ea1537e5c055ad4546f9e9b73181b6bc4eeebfd0119eec36a24ba6'
PROTOCOL_SHA = '2be63ef4fd10d1d30aa14b496ec238a1f06e0391c151937e484daa765c3d5ee4'
BUDGET, RSS_LIMIT, RANKS = 2**41, 4_000_000_000, (35, 100, 35)
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
VERSIONS = {'numpy': '2.4.6', 'flopscope': '0.12.0', 'whestbench': '0.16.0'}
FUNCTIONS = (
    '_cp3_bind_arena_views', 'gaussian_raw', '_square_schedule_reference',
    '_square_schedule', '_square_product', '_packed_frame',
    '_packed_product_cached', '_packed_product_3', '_packed_product_4',
    '_packed_product_5', '_mm', '_resident_gram512',
)
GLOBALS = (
    '_WINOGRAD', '_PACKED_ELEMENTS', '_PACKED_TILE', '_PACKED_PUSH',
    '_PACKED_READ', '_PACKED_SQUARE', '_PACKED_SQUARE4', '_PACKED_SQUARE5',
    '_PACKED_SQUARE6', '_SQUARE_COSTS', '_SQUARE_EQUAL_DEPTHS',
    '_SQUARE_CROSS_DEPTHS',
)
FIXED = {
    PARENT: PARENT_SHA,
    BD / 'native_v2.py': NATIVE_SHA,
    BD / 'qualify_native_v2.py': VERIFIER_SHA,
    HERE / 'watch.py': '736125d779a8eb9e1a530f7a3047276be585f8be09f47a4c48be0039e127a9d1',
    STORE / 'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    SITE / 'flopscope/stats/_norm.py': 'f23e49201c8e09e2785814b2b31663cda1c6bb217a57a211532e42fc9fe3b65d',
    SITE / 'flopscope/stats/_erf.py': '4cdd75fdb85981fc385417082985478566b18e19fa67ebb7d41225b5efaaa3bd',
    BR / 'native.py': '8e5af12f27aa28ae754fbcd2a10259ba96ce4ceb183c9084b84a23ce28ec876d',
    BR / 'watch.py': '1a72e56e9d6c642b2c3ec7f880cd8406fc873b51bcdb329d2076bed46cc5b599',
    BR / 'PROTOCOL.md': '2933bf2e390440b34b60f64c93cffc83dfb6845cd2f063ec4fea95368bb8c664',
    BR / 'run_null_v1/result.json': REFERENCE_SHA,
    BR / 'run_null_v1/completion.json': '315a134c516dc3f44a243446c130306641a244dc0ff5d305604457ab5910bc48',
    BR / 'watch_null_v1/result.json': WATCH_REFERENCE_SHA,
    BR / 'watch_null_v1/completion.json': '6b815682a31c332ad79114f3cdceebc60ddbd01bb4f0b43b20732d993ddbdfcd',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def extracted_source():
    """Compile only these last nodes later; retain their original source bytes."""
    source = PARENT.read_text()
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = node
    selected = [(name, functions[name]) for name in FUNCTIONS]
    selected += [(name, assignments[name]) for name in GLOBALS]
    values = {name: ast.literal_eval(assignments[name].value) for name in GLOBALS}
    require(values['_WINOGRAD'] is True and values['_PACKED_ELEMENTS'] == 292225024
            and values['_PACKED_TILE'] == 8192, 'Frozen matrix-only setup')
    require(values['_SQUARE_EQUAL_DEPTHS'][15][31] == 5, 'Fixed 512x1024x512 D5 dispatch')
    records, segments = {}, []
    for name, node in sorted(selected, key=lambda item: item[1].lineno):
        require(not isinstance(node, ast.FunctionDef) or not node.decorator_list, 'No extracted decorators')
        segment = ast.get_source_segment(source, node)
        require(segment is not None, 'Complete literal source segment')
        records[name] = dict(line=node.lineno, end_line=node.end_lineno,
                             sha256=hashlib.sha256(segment.encode()).hexdigest())
        segments.append(segment)
    bundle = '\n\n'.join(segments) + '\n'
    require(set(records) == set(FUNCTIONS + GLOBALS), 'Exact helper/global roster')
    return bundle, records


def authenticate(args):
    """Source/JSON/hash only, before numerical imports or array decoding."""
    pins = {}
    def pin(path, expected=None):
        p = Path(path).resolve()
        require(any(p.is_relative_to(root.resolve()) for root in (NEW, STORE, OFFPY)), 'Scoped binding: ' + str(p))
        digest = sha(p)
        require(expected is None or digest == expected, 'Binding mismatch: ' + str(p))
        require(str(p) not in pins or pins[str(p)] == digest, 'Conflicting binding')
        pins[str(p)] = digest
        return digest
    require(args.protocol_sha256 == PROTOCOL_SHA, 'Frozen prospective protocol')
    require(isinstance(args.source_sha256, str) and len(args.source_sha256) == 64, 'Reviewed source SHA required')
    pin(Path(__file__), args.source_sha256)
    pin(HERE / 'PROTOCOL.md', PROTOCOL_SHA)
    for path, digest in FIXED.items():
        pin(path, digest)
    native = module('g0_preserved_native_metadata', BD / 'native_v2.py')
    verifier = module('g0_preserved_meter_verifier', BD / 'qualify_native_v2.py')
    for path, digest in native.PINS.items():
        pin(path, digest)
    verifier.verify_meter(pin)
    manifest = read(STORE / 'manifest.json')['roles']['observer_fit']['row_manifests']
    rows = {}
    for rank in RANKS[:2]:
        matches = [x for x in manifest if x['role_rank'] == rank]
        require(len(matches) == 1, 'Unique canonical rank')
        entry = matches[0]
        rp = (STORE / entry['path']).resolve()
        pin(rp, entry['sha256'])
        row = read(rp)
        require(row['role_rank'] == rank and row['role'] == 'observer_fit', 'Canonical row frame')
        wp = rp.parent / 'weights.npy'
        info = row['files']['weights.npy']
        require(info['dtype'] == '<f4' and info['shape'] == [16, 1024, 1024], 'Fixed input file schema')
        pin(wp, info['sha256'])
        rows[rank] = dict(weights=str(wp), weight_sha256=info['sha256'], row=str(rp), row_sha256=entry['sha256'])
    ref = read(BR / 'run_null_v1/result.json')
    completion = read(BR / 'run_null_v1/completion.json')
    require(ref['passed'] and ref['hashes_unchanged'] and not ref['truth_accessed']
            and ref['candidate_sha256'] == PARENT_SHA and ref['ranks'] == [35, 100]
            and completion == dict(passed=True, result_sha256=REFERENCE_SHA), 'Sealed native null reference')
    require(Path(ref['source']).resolve() == BR / 'build_probes_v1/null/estimator.py', 'Exact null source identity')
    pin(ref['source'], PARENT_SHA)
    require(ref['source_bindings'][str(BD / 'native_v2.py')] == NATIVE_SHA, 'Reference lifecycle source')
    for row in rows.values():
        require(ref['source_bindings'][row['weights']] == row['weight_sha256']
                and ref['source_bindings'][row['row']] == row['row_sha256'], 'Reference uses identical fixed weights')
    require(len(ref['trials']) == 3 and [t['rank'] for t in ref['trials']] == list(RANKS), 'Complete native reference sequence')
    # Reauthenticate its complete saved artifacts without opening P/gradient NPZs
    # from the broader bridge provenance.  The entire reference JSON is fixed.
    for path, digest in ref['artifacts'].items():
        require(Path(path).resolve().parent == BR / 'run_null_v1', 'Only sealed reference artifacts')
        pin(path, digest)
    for call, trial in enumerate(ref['trials']):
        require(trial['call'] == call and trial['output'] == str(BR / 'run_null_v1' / f'call_{call:02d}_rank{RANKS[call]}.npy'), 'Reference output identity')
        require(ref['artifacts'][trial['output']] == trial['output_sha256'], 'Reference output seal')
        verifier.guard(trial)
    watch = read(BR / 'watch_null_v1/result.json')
    wc = read(BR / 'watch_null_v1/completion.json')
    require(watch['passed'] and watch['completed'] and watch['sources_unchanged']
            and not watch['terminated'] and watch['child_exit_code'] == 0
            and watch['child_result_sha256'] == REFERENCE_SHA
            and wc['passed'] and wc['completed'] and wc['result_sha256'] == WATCH_REFERENCE_SHA,
            'Watched native reference completed')
    bundle, records = extracted_source()
    return pins, pin, native, verifier, rows, ref, bundle, records


def guard(native, record):
    native.guard(record)
    require(0 < record['peak_rss_bytes'] < RSS_LIMIT, 'Strict 4GB process RSS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    sys.dont_write_bytecode = True
    require(Path(sys.prefix).resolve() == OFFPY.resolve() and 'anaconda3' not in sys.executable, 'OFFPY only')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Required nice10 launch')
    require(all(os.environ.get(k) == '1' for k in BLAS), 'Fixed BLAS launch variables')
    require(all(importlib.metadata.version(k) == v for k, v in VERSIONS.items()), 'Frozen library versions')
    require('torch' not in sys.modules and 'numpy' not in sys.modules, 'No prior numerical imports')
    out = HERE / 'acquisition_v1'
    require(not out.exists(), 'Exclusive one-shot acquisition')
    pins, pin, native, verifier, rows, reference, bundle, extraction = authenticate(args)
    out.mkdir()
    result = dict(schema='current_d50_initial_gram_witness_v1', passed=False, completed=False,
        hashes_unchanged=False, started_utc=datetime.now(timezone.utc).isoformat(),
        source_bindings=pins, input_bindings={row['weights']: row['weight_sha256'] for row in rows.values()},
        artifacts={}, trials=[], input_preparation=[], diagnostics=[], ranks=list(RANKS),
        truth_accessed=False, weights_read=False, weight_layers_read=[0], full_card_calls=0,
        P_accessed=False, cstar_accessed=False, score_accessed=False, no_fit=True,
        initialization_once=True, complete_estimator_initialization_claimed=False,
        native_reference_sha256=REFERENCE_SHA, helper_source_records=extraction,
        helper_bundle_sha256=hashlib.sha256(bundle.encode()).hexdigest(),
        versions=VERSIONS, blas_environment={k: os.environ[k] for k in BLAS},
        input_preparation_in_participant_bill=False, candidate_state_reset=False,
        reset_scope='Public diagnostic history only, after saved closed bills',
        gram_convention='Native FP32 three-product Gram widened to FP64; Cpre and Bpre alias',
        initialization_scope='Original matrix arena and push/read views only; lazy square/cache work retained')
    native.write(out / 'started.json', result)
    try:
        # The only decoded reference is the already authenticated served output.
        reference_bytes = {t['rank']: verifier.payload(t['output'], t['output_bytes_sha256'], pin)[:1024 * 4]
                           for t in reference['trials']}
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits, threadpool_info
        require(np.__version__ == VERSIONS['numpy'] and 'torch' not in sys.modules, 'Actual NumPy/no MPS')
        flops.configure(symmetry_warnings=False)
        env = {'fnp': fnp, 'flops': flops}
        exec(compile(bundle, str(PARENT) + ':isolated-last-helpers', 'exec'), env)
        env['_SCRATCH'] = None
        env['_PACKED_FRAME_CACHE'] = {}
        require(env['_square_schedule'](512, 1024, 512) == (5, (512, 1024, 512)), 'Exact D5, no padding')
        first = None
        with threadpool_limits(limits=1):
            result['threadpools'] = threadpool_info()
            require(all(p['num_threads'] == 1 for p in result['threadpools'] if p.get('user_api') == 'blas'), 'Observed BLAS limits')
            result['empty_threadpool_inventory_is_unknown_instrumentation'] = not result['threadpools']
            for call, rank in enumerate(RANKS):
                weights = np.load(rows[rank]['weights'], mmap_mode='r', allow_pickle=False)
                require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32, 'Canonical weight layout')
                w0 = weights[0]
                result['weights_read'] = True
                require(np.isfinite(w0).all(), 'Finite layer0 weights only')
                started = time.monotonic()
                with flops.BudgetContext(flop_budget=BUDGET, wall_time_limit_s=120, quiet=True) as input_ctx:
                    owned = fnp.array(w0, dtype=fnp.float32)
                prep = dict(call=call, rank=rank, **native.metrics(input_ctx, started),
                    charged_to_participant=False, layers=1, layer_indices=[0], context_closed=True,
                    weight_file_sha256=rows[rank]['weight_sha256'],
                    official_fnp_type=isinstance(owned, fnp.ndarray) and owned.dtype == np.float32 and owned.shape == (1024, 1024),
                    exact_bytes=owned.tobytes() == w0.tobytes(),
                    independent_weight_buffers=not np.may_share_memory(np.asarray(owned), weights))
                prep['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                input_bill = out / f'call_{call:02d}_input_bill.json'
                native.write(input_bill, prep)
                result['input_preparation'].append(prep)
                result['artifacts'][str(input_bill)] = sha(input_bill)
                input_ref = weakref.ref(input_ctx)
                reset = native.reset_saved_history(flops, input_ctx, input_bill)
                del input_ctx
                gc.collect()
                reset.update(call=call, rank=rank, context_released=input_ref() is None,
                             complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
                input_reset = out / f'call_{call:02d}_input_reset.json'
                native.write(input_reset, reset)
                result['artifacts'][str(input_reset)] = sha(input_reset)
                guard(native, prep)
                require(prep['flops'] == 1024 * 1024 and prep['official_fnp_type']
                        and prep['exact_bytes'] and prep['independent_weight_buffers'], 'Exact separately paid owned input')
                require(reset['context_released'] and reset['closed_context_bill_unchanged_by_reset'], 'Input log lifecycle')
                del w0, weights
                started = time.monotonic()
                with flops.BudgetContext(flop_budget=BUDGET, wall_time_limit_s=120, quiet=True) as ctx:
                    setup_started = time.monotonic()
                    if env['_SCRATCH'] is None:
                        arena = fnp.empty((env['_PACKED_ELEMENTS'],), dtype=fnp.float32)
                        env['_SCRATCH'] = [arena,
                            env['_cp3_bind_arena_views'](arena, env['_PACKED_PUSH']),
                            env['_cp3_bind_arena_views'](arena, env['_PACKED_READ']), None, None, None, None]
                    setup = dict(flops=int(ctx.flops_used), operation_count=len(ctx.op_log),
                                 wall_s=time.monotonic() - setup_started)
                    w32 = fnp.asarray(owned, dtype=fnp.float32)
                    w64 = fnp.asarray(owned, dtype=fnp.float64)
                    mu = w64.T @ fnp.zeros(1024, dtype=fnp.float64)
                    G0 = env['_resident_gram512'](w32.T, w32).astype(fnp.float64)
                    Cpre = Bpre = G0
                    raw = env['gaussian_raw'](mu, fnp.diag(Cpre))
                    served = fnp.asarray(fnp.maximum(raw[4], 0.0), dtype=fnp.float32)
                    # Materialization stays inside the paid context.  NPY file I/O
                    # below uses only these copied bytes, never new FNP arithmetic.
                    gram_bytes, mean_bytes = G0.tobytes(), served.tobytes()
                trial = dict(call=call, rank=rank, **native.metrics(ctx, started),
                    includes_matrix_initialization=(call == 0), matrix_arena_setup=setup,
                    formula_and_materialization_flops=int(ctx.flops_used) - setup['flops'],
                    gram_shape=list(G0.shape), gram_dtype=str(G0.dtype),
                    mean_shape=list(served.shape), mean_dtype=str(served.dtype),
                    Cpre_Bpre_same_object=Cpre is Bpre,
                    input_bill=str(input_bill), input_bill_sha256=sha(input_bill),
                    input_reset_receipt=str(input_reset), input_reset_sha256=sha(input_reset),
                    lifecycle_receipt=str(out / f'call_{call:02d}_lifecycle.json'))
                gram = np.frombuffer(gram_bytes, dtype=np.float64).reshape(1024, 1024)
                mean = np.frombuffer(mean_bytes, dtype=np.float32).reshape(1024)
                for name, values, data in (('G0', gram, gram_bytes), ('mean', mean, mean_bytes)):
                    path = out / f'call_{call:02d}_rank{rank}_{name}.npy'
                    with path.open('xb') as f:
                        np.save(f, values, allow_pickle=False)
                    result['artifacts'][str(path)] = sha(path)
                    trial[name] = dict(path=str(path), sha256=sha(path), bytes_sha256=hashlib.sha256(data).hexdigest())
                trial['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                result['trials'].append(trial)
                bill = out / f'call_{call:02d}_bill.json'
                native.write(bill, trial)
                result['artifacts'][str(bill)] = sha(bill)
                context_ref = weakref.ref(ctx)
                lifecycle = native.reset_saved_history(flops, ctx, bill)
                del ctx
                gc.collect()
                lifecycle.update(call=call, rank=rank, context_released=context_ref() is None,
                                 complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
                lifecycle_path = Path(trial['lifecycle_receipt'])
                native.write(lifecycle_path, lifecycle)
                result['artifacts'][str(lifecycle_path)] = sha(lifecycle_path)
                # All immutable output/bill/reset artifacts precede these gates.
                diagnostic = dict(call=call, rank=rank, G0_finite=bool(np.isfinite(gram).all()),
                    minimum_diagonal=float(np.min(np.diag(gram))),
                    served_finite_nonnegative=bool(np.isfinite(mean).all() and (mean >= 0).all()),
                    native_layer0_byte_exact=mean_bytes == reference_bytes[rank],
                    copied_cross_block_byte_exact=gram[:512, 512:].T.tobytes() == gram[512:, :512].tobytes(),
                    frame_shapes=[list(key[1:]) for key in env['_PACKED_FRAME_CACHE']],
                    only_D5_square_views_used=env['_SCRATCH'][5] is not None and all(env['_SCRATCH'][j] is None for j in (3, 4, 6)),
                    repeat_G0_mean_byte_exact=(None if call != 2 else first == (gram_bytes, mean_bytes)))
                result['diagnostics'].append(diagnostic)
                native.write(out / f'call_{call:02d}_checks.json', diagnostic)
                check_path = out / f'call_{call:02d}_checks.json'
                result['artifacts'][str(check_path)] = sha(check_path)
                guard(native, trial)
                require(lifecycle['context_released'] and lifecycle['closed_context_bill_unchanged_by_reset'], 'Paid context log lifecycle')
                require(trial['gram_dtype'] == 'float64' and trial['mean_dtype'] == 'float32'
                        and trial['gram_shape'] == [1024, 1024] and trial['mean_shape'] == [1024]
                        and trial['Cpre_Bpre_same_object'], 'Exact served shape/dtype/alias')
                require(diagnostic['G0_finite'] and diagnostic['minimum_diagonal'] >= 1e-10
                        and diagnostic['served_finite_nonnegative'] and diagnostic['native_layer0_byte_exact']
                        and diagnostic['copied_cross_block_byte_exact'] and diagnostic['only_D5_square_views_used']
                        and diagnostic['frame_shapes'] == [[5, 512, 1024, 512]], 'Fixed Gram/native endpoint gates')
                if call == 0:
                    first = (gram_bytes, mean_bytes)
                if call == 2:
                    require(diagnostic['repeat_G0_mean_byte_exact'], 'Exact repeated Gram and mean')
                print(json.dumps(dict(call=call, rank=rank, flops=trial['flops'], passed=True)), flush=True)
                del owned, w32, w64, mu, G0, Cpre, Bpre, raw, served, gram, mean, gram_bytes, mean_bytes
                gc.collect()
        require('torch' not in sys.modules, 'No MPS imports or calls')
        require(all(sha(path) == digest for path, digest in pins.items()), 'Inputs and sources unchanged')
        require(all(sha(path) == digest for path, digest in result['artifacts'].items()), 'Artifacts unchanged')
        result.update(passed=True, completed=True, hashes_unchanged=True,
            cold_minus_repeat_subroutine_flops=result['trials'][0]['flops'] - result['trials'][2]['flops'],
            all_native_layer0_bytes_exact=True, repeat_Gram_mean_byte_exact=True)
    except BaseException:
        result.update(passed=False, completed=False, hashes_unchanged=False, error=traceback.format_exc())
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    native.write(out / 'result.json', result)
    native.write(out / 'completion.json', dict(passed=result['passed'], completed=result['completed'],
                                               result_sha256=sha(out / 'result.json')))
    print(json.dumps(dict(passed=result['passed'], completed=result['completed'], error=result.get('error'))), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
