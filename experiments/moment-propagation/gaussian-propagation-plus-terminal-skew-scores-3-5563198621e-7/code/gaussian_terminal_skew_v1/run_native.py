"""Inert --execute fixed Gaussian/skew native35/100/35 acquisition; NO truths.

Use separate fresh nice10 OFFPY processes, one numerical worker at a time.
Complete predict/init/materialization bills are saved before public diagnostic
reset. The input FNP copy context is separately recorded, not participant work.
"""
import argparse
import gc
import json
import signal
import time
import traceback
import common as h


def fixtures():
    path = h.STORE / 'manifest.json'
    h.require(h.sha(path) == '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
              'Canonical source manifest')
    bindings = {str(path): h.sha(path)}
    entries = json.loads(path.read_text())['roles']['observer_fit']['row_manifests']
    rows = {}
    for rank in (35, 100):
        item = next(x for x in entries if int(x['role_rank']) == rank)
        rowpath = (h.STORE / item['path']).resolve()
        h.require(rowpath.is_relative_to(h.STORE.resolve()), 'Canonical row location')
        h.require(h.sha(rowpath) == item['sha256'], 'Row manifest seal')
        row = json.loads(rowpath.read_text())
        weight = rowpath.parent / 'weights.npy'
        h.require(h.sha(weight) == row['files']['weights.npy']['sha256'], 'Canonical weight seal')
        bindings[str(rowpath)], bindings[str(weight)] = h.sha(rowpath), h.sha(weight)
        rows[rank] = dict(weights=str(weight), mlp_seed=int(row['mlp_seed']))
    return rows, bindings


def small_gate(bindings):
    path, seal = h.HERE / 'small/result.json', h.HERE / 'small/completion.json'
    result, completed = json.loads(path.read_text()), json.loads(seal.read_text())
    h.require(completed['passed'] and completed['completed'] and completed['result_sha256'] == h.sha(path),
              'Fixed tiny completion seal')
    h.require(result['passed'] and result['completed'] and result['sources_unchanged'], 'Fixed tiny qualification')
    h.require(result['source_bindings'] == bindings, 'Exactly the source set qualified on the tiny fixture')
    h.require(result['candidate_sha256'] == h.sha(h.HERE / 'estimator.py'), 'Tiny candidate identity')
    return {str(path): h.sha(path), str(seal): h.sha(seal)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--role', choices=('gaussian', 'skew'))
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return 0
    h.require(args.role is not None, 'Fixed role required')
    environment, bindings = h.runtime(), h.sources()
    bindings.update(small_gate(bindings))
    rows, fixture_bindings = fixtures()
    bindings.update(fixture_bindings)
    out = h.HERE / ('native_' + args.role)
    out.mkdir()
    result = dict(passed=False, completed=False, role=args.role, sequence=[35, 100, 35],
                  candidate_sha256=h.sha(h.HERE / 'estimator.py'), source_bindings=bindings,
                  fixture_rows=rows, environment=environment, truth_accessed=False,
                  initialization_once=True, repeat_excluded_from_score=True,
                  official_fnp_input=True, input_preparation_in_participant_bill=False,
                  production_residual_qualified=False, started_utc=h.utc(),
                  trials=[], input_preparation=[], artifacts={})
    h.write(out / 'started.json', result)
    try:
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_info, threadpool_limits
        from whestbench.domain import MLP
        from whestbench.seeds import derive_estimator_seed
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        module = h.load_estimator()
        constructor = module.GaussianEstimator if args.role == 'gaussian' else module.Estimator
        estimator, first_bytes, first_flops = None, None, None
        with threadpool_limits(limits=1):
            result['threadpools'] = threadpool_info()
            h.require(all(x['num_threads'] == 1 for x in result['threadpools']
                          if x.get('user_api') == 'blas'), 'BLAS1')
            for call, rank in enumerate((35, 100, 35)):
                h.alarm(130)
                input_start = time.monotonic()
                weights = np.load(rows[rank]['weights'], mmap_mode='r', allow_pickle=False)
                h.require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32
                          and np.isfinite(weights).all(), 'Official weight array contract')
                input_error = None
                with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as input_ctx:
                    try:
                        owned = [fnp.array(w, dtype=fnp.float32) for w in weights]
                    except BaseException:
                        input_error = traceback.format_exc()
                prep = dict(call=call, rank=rank, **h.metrics(input_ctx, input_start),
                            charged_to_participant=False, error=input_error, context_closed=True)
                if input_error is None:
                    prep.update(exact_bytes=all(w.tobytes() == weights[i].tobytes() for i, w in enumerate(owned)),
                                official_type=all(isinstance(w, fnp.ndarray) and w.dtype == fnp.float32 for w in owned),
                                independent_storage=all(not np.may_share_memory(np.asarray(w), weights) for w in owned))
                prep['peak_rss_bytes'] = h.rss()
                result['input_preparation'].append(prep)
                path = out / f'call_{call:02d}_input.json'
                h.write(path, prep)
                result['artifacts'][str(path)] = h.sha(path)
                ref, release = h.reset_closed(flops, input_ctx, path)
                del input_ctx
                gc.collect()
                release['context_released'] = ref() is None
                rpath = out / f'call_{call:02d}_input_release.json'
                h.write(rpath, release)
                result['artifacts'][str(rpath)] = h.sha(rpath)
                h.require(input_error is None, 'Input preparation failed')
                h.guard(prep)
                h.require(prep['exact_bytes'] and prep['official_type'] and prep['independent_storage'], 'Owned FNP byte identity')
                h.require(prep['flops'] == 16 * 1024**2 and release['context_released'], 'Input tariff/lifetime')
                mlp = MLP(width=1024, depth=16, weights=owned,
                          seed=int(derive_estimator_seed(rows[rank]['mlp_seed'])), name=f'gaussian-terminal-{rank}')
                del weights, owned
                start, error, served, materialized, ctx = time.monotonic(), None, None, None, None
                try:
                    with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=120, quiet=True) as ctx:
                        if estimator is None:
                            estimator = constructor()
                        served = estimator.predict(mlp, h.BUDGET)
                        h.require(isinstance(served, np.ndarray) and served.shape == (16, 1024)
                                  and served.dtype == np.float32, 'Concrete served float32 contract')
                        materialized = materialise_predictions(served, depth=16, width=1024)
                except BaseException:
                    error = traceback.format_exc()
                if ctx is None:
                    attempted = dict(call=call, rank=rank, error=error, context_created=False,
                                     wall_s=time.monotonic() - start, peak_rss_bytes=h.rss())
                    result['trials'].append(attempted)
                    path_failed = out / f'call_{call:02d}_attempt.json'
                    h.write(path_failed, attempted)
                    result['artifacts'][str(path_failed)] = h.sha(path_failed)
                    raise RuntimeError('Participant context could not be created')
                trial = dict(call=call, rank=rank, **h.metrics(ctx, start), error=error,
                             includes_initialization=call == 0, input_receipt=str(path),
                             multiplier=max(.1, float(ctx.flops_used) / h.BUDGET))
                # Preserve the complete attempt even if later serialization fails.
                result['trials'].append(trial)
                attempt = out / f'call_{call:02d}_attempt.json'
                h.write(attempt, trial)
                result['artifacts'][str(attempt)] = h.sha(attempt)
                if error is None:
                    original, returned = np.asarray(served), np.asarray(materialized)
                    trial['served_materialized_byte_exact'] = original.tobytes() == returned.tobytes()
                    trial['output_contract'] = bool(returned.shape == (16, 1024) and returned.dtype == np.float32
                                                    and np.isfinite(returned).all() and (returned >= 0).all())
                    # A base NumPy byte view, never a dtype cast or prediction repair.
                    values = np.frombuffer(returned.tobytes(), dtype=np.float32).reshape(16, 1024)
                    output = out / f'call_{call:02d}_rank{rank}.npy'
                    with output.open('xb') as f:
                        np.save(f, values, allow_pickle=False)
                    result['artifacts'][str(output)] = h.sha(output)
                    trial.update(output=str(output), output_sha256=h.sha(output),
                                 diagnostics=estimator.last_diagnostics)
                trial['peak_rss_bytes'] = h.rss()
                bill = out / f'call_{call:02d}_bill.json'
                h.write(bill, trial)
                result['artifacts'][str(bill)] = h.sha(bill)
                print(json.dumps(h.safe(trial)), flush=True)
                ref, release = h.reset_closed(flops, ctx, bill)
                del ctx
                gc.collect()
                release['context_released'] = ref() is None
                rpath = out / f'call_{call:02d}_release.json'
                h.write(rpath, release)
                result['artifacts'][str(rpath)] = h.sha(rpath)
                h.require(error is None, 'Participant failed; attempted bill preserved')
                h.guard(trial)
                h.require(trial['served_materialized_byte_exact'] and trial['output_contract'], 'Output identity/contract')
                h.require(trial['diagnostics']['fp32_matrix_products'] == (30 if args.role == 'gaussian' else 59),
                          'Fixed full-width matrix-product count')
                h.require(release['context_released'], 'Participant context released')
                if call == 0:
                    first_bytes, first_flops = values.tobytes(), trial['flops']
                if call == 2:
                    h.require(values.tobytes() == first_bytes and trial['flops'] == first_flops,
                              'Repeat bytes/bill; constructors contain no numeric initialization')
                del mlp, served, materialized, original, returned, values
                gc.collect()
                h.require(h.rss() < 8_000_000_000, 'End-of-call lifetime RSS')
                signal.setitimer(signal.ITIMER_REAL, 0)
        h.verify(bindings)
        h.verify(result['artifacts'])
        result.update(passed=True, completed=True, all_bindings_unchanged=True, initialization_flops=0)
    except BaseException:
        result['error'] = traceback.format_exc()
    signal.setitimer(signal.ITIMER_REAL, 0)
    h.finish(out, result)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
