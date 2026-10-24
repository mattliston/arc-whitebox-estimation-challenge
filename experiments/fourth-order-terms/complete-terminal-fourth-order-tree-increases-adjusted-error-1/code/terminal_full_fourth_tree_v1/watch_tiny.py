"""One-shot stdlib-only supervisor for the fixed n=4 TREE certificate.

Inert without --run. A completed TLv2 validation is required, regardless of
its scientific outcome. This launches only the tiny checker, never a model,
fit, scorer or native acquisition. No numerical library or target is read.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
TL = HERE.parent / 'current_d50_terminal_linear_v2'
OFFPY = HERE.parent.parent / 'data_gate/.venv/bin/python'
HELPER = TL / 'watch_fit.py'
HELPER_SHA = 'b2d3c1334bd642306eea4f0ee12d63fc9e16f8dfc557e36e0db00a8030a6eebd'
CHECKER_SHA = 'd7ea9898f55555591f3d23cca0d04c0a412b38bca879f99d3f02f36866dd668c'
BUILDER_SHA = 'ae8f90bd3767f89e157b32dd75817b0c65e055091667ba20d2476e6a81a0c2f8'
PROTOCOL_SHA = 'f4bf2907fa1973b6076d6f485cd2436d6b5bf13cfe26077653e52af3c3e59dad'
BUILD_SHA = 'a94d6109b2ec176a9592a6593b3a6f9ebc5bf8f37f953b78f59dd992b39bda00'
TL_JUDGE_SHA = 'd755605c7a1773cf3fe740ee7a494435fc67efeb7377fc0f442b9e00f9277c9e'
TL_PROTOCOL_SHA = 'febed4bdccc7c031cb846e6d97f695536ca98d3bef10d07de48f1418a91b6a58'
WALL, RSS, SWAP = 30.0, 1_000_000_000, 512 * 1024 ** 2
PERIOD, MAX_INTERVAL = .2, .5
BLAS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def authenticate(args):
    require(args.checker_sha256 == CHECKER_SHA and args.builder_sha256 == BUILDER_SHA
            and args.protocol_sha256 == PROTOCOL_SHA, 'Fixed reviewed tiny sources')
    sources = {str(Path(__file__).resolve()): args.source_sha256,
               str(HERE / 'check_tiny.py'): CHECKER_SHA,
               str(HERE / 'build.py'): BUILDER_SHA,
               str(HERE / 'PROTOCOL.md'): PROTOCOL_SHA, str(HELPER): HELPER_SHA,
               str(TL / 'judge_terminal.py'): TL_JUDGE_SHA,
               str(TL / 'PROTOCOL.md'): TL_PROTOCOL_SHA}
    for path, digest in sources.items():
        require(sha(path) == digest, 'Reviewed source changed: ' + path)
    builder = module(HERE / 'build.py', 'tree_tiny_source_builder')
    child_sources = builder.authenticate(BUILDER_SHA, PROTOCOL_SHA)
    child_sources[str(HERE / 'check_tiny.py')] = CHECKER_SHA
    sources.update(child_sources)
    inputs = {}

    def pin(path, expected=None):
        digest = sha(path)
        require(expected is None or digest == expected, 'Input seal changed: ' + str(path))
        inputs[str(path)] = digest
        return digest

    build_path, build_completion = HERE / 'build_v1/build.json', HERE / 'build_v1/completion.json'
    pin(build_path, BUILD_SHA); pin(build_completion)
    build, seal = read(build_path), read(build_completion)
    require(all(build.get(k) is True for k in ('passed', 'completed', 'hashes_unchanged',
            'sources_unchanged', 'exact_source_inverse', 'only_enable_literal_differs'))
            and seal.get('passed') is True and seal.get('completed') is True
            and seal.get('result_sha256') == BUILD_SHA, 'Completed source-only build seal')
    require(build['source_bindings'] == {k: v for k, v in child_sources.items()
            if k != str(HERE / 'check_tiny.py')}, 'Build and checker source identity')
    for path, digest in build['artifacts'].items():
        pin(path, digest)  # Source files only; never import the candidates.

    score_path, score_completion = TL / 'terminal_score_v1/result.json', TL / 'terminal_score_v1/completion.json'
    pin(score_path, args.terminal_score_sha256); pin(score_completion)
    score, seal = read(score_path), read(score_completion)
    require(score.get('schema') == 'current_d50_terminal_linear_native40_score_v1'
            and all(score.get(k) is True for k in ('completed', 'integrity_passed',
                'hashes_unchanged', 'all_predictions_verified_before_truth',
                'all_earlier15_outputs_byte_exact', 'all_own_complete_bills_verified'))
            and len(score.get('records', [])) == 40
            and seal.get('completed') is True
            and seal.get('result_sha256') == args.terminal_score_sha256
            and seal.get('passed') is score.get('passed'), 'Completed fixed TLv2 validation required')
    require(score['source_bindings'][str(TL / 'judge_terminal.py')] == TL_JUDGE_SHA
            and score['source_bindings'][str(TL / 'PROTOCOL.md')] == TL_PROTOCOL_SHA,
            'TLv2 validation source identity')
    # Deliberately do not require score.passed or advancement and do not open
    # any of its target, prediction, fit or numerical payload files.
    require(not any(name in sys.modules for name in ('numpy', 'torch', 'flopscope')),
            'Supervisor admission must remain stdlib only')
    return sources, inputs, child_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    for name in ('source', 'checker', 'builder', 'protocol', 'terminal-score'):
        parser.add_argument('--' + name + '-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    require(all(re.fullmatch(r'[0-9a-f]{64}', v or '') for k, v in vars(args).items()
                if k != 'run'), 'Every reviewed SHA256 is required')
    started = time.monotonic()
    sources, inputs, child_sources = authenticate(args)
    helper = module(HELPER, 'tree_tiny_resource_helpers')
    require(OFFPY.is_file(), 'Fixed OFFPY interpreter missing')
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    require(priority >= 10, 'nice10 watcher and child required')
    environment = {key: '1' for key in BLAS}
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTORCH_ENABLE_MPS_FALLBACK='0')
    require(all(os.environ.get(k) == v for k, v in environment.items()), 'Explicit BLAS1/no-bytecode launch')
    out, child_out = HERE / 'tiny_watch_v1', HERE / 'tiny_v1'
    require(not child_out.exists(), 'Tiny checker has already been attempted')
    out.mkdir(exist_ok=False)
    command = [str(OFFPY), '-B', '-u', str(HERE / 'check_tiny.py'), '--run',
               '--source-sha256', CHECKER_SHA, '--builder-sha256', BUILDER_SHA,
               '--protocol-sha256', PROTOCOL_SHA]
    result = dict(schema='terminal_full_fourth_tree_tiny_watch_v1', passed=False,
        completed=False, hashes_unchanged=False, sources_unchanged=False,
        source_bindings=sources, input_bindings=inputs, artifacts={}, command=command,
        prerequisites={'terminal_score_sha256': args.terminal_score_sha256, 'build_sha256': BUILD_SHA},
        child_output=str(child_out), nice=priority, inherited_environment=environment,
        wall_limit_s=WALL, combined_rss_limit_bytes=RSS, swap_growth_limit_bytes=SWAP,
        requested_sample_period_s=PERIOD, sample_interval_limit_s=MAX_INTERVAL,
        maximum_sample_interval_s=0.0, maximum_sampled_rss_bytes=0, maximum_swap_growth_bytes=0.0,
        samples=0, truth_accessed=False, numerical_imports=False, MPS_used=False,
        scientific_advancement=False, native_or_score_admitted=False,
        limitations=['RSS and swap are sampled, not continuous peak guarantees.',
                     'Swap growth is system-wide, not attributed to this child.',
                     'PASS is a tiny algebra/resource certificate, not a candidate score.'],
        started_utc=datetime.now(timezone.utc).isoformat())
    child, previous = None, {}
    last = time.monotonic()

    def interrupt(signum, frame):
        raise RuntimeError('Watcher interrupted by signal ' + str(signum))

    def observe(resources=None):
        nonlocal last
        item = helper.memory_snapshot(child.pid if child is not None else os.getpid())
        sampled = time.monotonic()
        interval, last = sampled - last, sampled
        swap = helper.swap_state()
        growth = swap['used_bytes'] - result['before_swap']['used_bytes']
        item.update(elapsed_s=time.monotonic() - started, interval_s=interval,
                    swap=swap, swap_growth_bytes=growth)
        result['samples'] += 1
        result['maximum_sampled_rss_bytes'] = max(result['maximum_sampled_rss_bytes'], item['combined_rss_bytes'])
        result['maximum_sample_interval_s'] = max(result['maximum_sample_interval_s'], interval)
        result['maximum_swap_growth_bytes'] = max(result['maximum_swap_growth_bytes'], growth)
        if resources is not None:
            resources.write(json.dumps(item, allow_nan=False) + '\n'); resources.flush()
        require(item['elapsed_s'] < WALL, 'Whole watcher/child30s limit')
        require(interval <= MAX_INTERVAL, 'Resource sampling gap exceeds .5 seconds')
        require(item['combined_rss_bytes'] < RSS, 'Combined watcher/child1GB limit')
        require(growth <= SWAP, 'System swap-growth limit')
        return item

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, interrupt)
        result['before_swap'] = helper.swap_state()
        write(out / 'started.json', result)
        with (out / 'stdout.log').open('xb') as stdout, (out / 'stderr.log').open('xb') as stderr, \
                (out / 'resources.jsonl').open('x') as resources:
            observe(resources)
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                env=os.environ.copy(), start_new_session=True)
            result['child_pid'] = result['child_process_group'] = child.pid
            while True:
                code = child.poll()  # Reap before checking for surviving group members.
                item = observe(resources)
                require(item['child_pids'] or code is not None or child.poll() is not None,
                        'Live child tree not observable')
                if code is not None:
                    result['child_exit_code'] = code
                    require(not item['child_pids'], 'Child left owned descendants alive')
                    break
                time.sleep(min(PERIOD, max(0.0, WALL - (time.monotonic() - started))))
        require(result['child_exit_code'] == 0, 'Tiny checker failed')
        rp, cp = child_out / 'result.json', child_out / 'completion.json'
        record, seal = read(rp), read(cp)
        result.update(child_result_sha256=sha(rp), child_completion_sha256=sha(cp))
        require(record.get('schema') == 'terminal_full_fourth_tree_tiny_v1'
                and all(record.get(k) is True for k in ('passed', 'completed', 'hashes_unchanged',
                    'sources_unchanged', 'exact_source_inverse', 'wick_factorials_verified',
                    'zero_source_passed', 'resource_gate_passed'))
                and seal.get('passed') is True and seal.get('completed') is True
                and seal.get('result_sha256') == result['child_result_sha256'], 'Tiny completion/algebra seal')
        require(record['source_bindings'] == child_sources and record['artifacts'] == {},
                'Exact tiny source/artifact roster')
        require(record['truth_accessed'] is False and record['model_calls'] == 0
                and record['MPS_used'] is False and record['candidate_files_created'] is False
                and record['full_cumulant_claimed'] is False
                and record['algebra_only_not_score_evidence'] is True
                and record['numerical_run_or_score_admitted'] is False, 'Algebra-only scope')
        require(len(record['checks']) == 58 and all(c['passed'] is True for c in record['checks'])
                and math.isfinite(record['wall_s']) and 0 <= record['wall_s'] < WALL
                and 0 < record['peak_rss_bytes'] < RSS, 'Complete tiny checks and worker resource guard')
        result.update(passed=True, completed=True, child_execution_completed=True,
                      whole_owned_tree_retired=True)
    except BaseException:
        result.update(passed=False, completed=False, hashes_unchanged=False, sources_unchanged=False,
                      error=traceback.format_exc())
        if child is not None:
            try:
                helper.stop_owned_group(child)
                result.update(whole_owned_tree_retired=True, child_exit_code=child.poll())
            except BaseException:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        try:
            for path in (out / 'started.json', out / 'stdout.log', out / 'stderr.log', out / 'resources.jsonl',
                         child_out / 'result.json', child_out / 'completion.json'):
                if path.is_file():
                    result['artifacts'][str(path)] = sha(path)
            for path, digest in {**sources, **inputs}.items():
                require(sha(path) == digest, 'Final source/input seal changed: ' + path)
            result.update(hashes_unchanged=True, sources_unchanged=True)
            if result['passed']:
                item = observe()  # After logs/source/receipt hashing; sealed resource log stays closed.
                require(not item['child_pids'], 'Owned descendants after final sealing')
                result['final_resource_sample_after_hashing'] = item
                require(sha(child_out / 'result.json') == result['child_result_sha256']
                        and sha(child_out / 'completion.json') == result['child_completion_sha256'],
                        'Child receipt changed during final verification')
        except BaseException:
            result.update(passed=False, completed=False, hashes_unchanged=False, sources_unchanged=False,
                          final_sealing_error=traceback.format_exc())
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        result.update(elapsed_s=time.monotonic() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write(out / 'result.json', result)
        write(out / 'completion.json', dict(passed=result['passed'], completed=result['completed'],
            hashes_unchanged=result['hashes_unchanged'], sources_unchanged=result['sources_unchanged'],
            result_sha256=sha(out / 'result.json')))
    print(json.dumps({k: result.get(k) for k in ('passed', 'completed', 'elapsed_s',
        'maximum_sampled_rss_bytes', 'maximum_sample_interval_s', 'maximum_swap_growth_bytes',
        'child_result_sha256', 'error', 'cleanup_error', 'final_sealing_error')}, sort_keys=True), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
