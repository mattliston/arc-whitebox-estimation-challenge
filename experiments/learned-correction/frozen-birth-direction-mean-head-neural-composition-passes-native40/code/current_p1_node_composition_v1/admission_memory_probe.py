"""One stdlib-only admission/GC measurement; never imports an estimator."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
NATIVE_SHA = 'fafc28732701aacfa70ada15d30ed083af76e91747bc01dc840ec8ba14d23d01'
WATCH_SHA = '7cf461c4c5e847a81c8b20b6c6c37f7f9169aee1d0d4507382e2842caed8d53b'
FAIL_SHA = '56046ed7ef0ea50c7af5e5e8f5cbe51a38181721bd753b38f5070477eda0f66d'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    assert sys.flags.optimize == 0 and sys.dont_write_bytecode
    assert os.getpriority(os.PRIO_PROCESS,0) >= 10
    assert sha(__file__) == args.source_sha256
    assert sha(HERE/'native.py') == NATIVE_SHA and sha(HERE/'watch_native.py') == WATCH_SHA
    assert sha(HERE/'native_terminal_watch_v1/result.json') == FAIL_SHA
    failed = json.loads((HERE/'native_terminal_watch_v1/result.json').read_text())
    assert failed['passed'] is False and failed['whole_owned_tree_retired']
    out = HERE/'admission_memory_v1'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    record = dict(passed=False,completed=False,source_sha256=args.source_sha256,
        failed_watch_sha256=FAIL_SHA,estimator_executed=False,numerical_imports=False,
        truth_accessed=False,source_only=True,started_utc=datetime.now(timezone.utc).isoformat())
    def guard(*unused):
        assert time.monotonic()-start < 60
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1_000_000_000
    def rss():
        guard()
        value = subprocess.check_output(['/bin/ps','-o','rss=','-p',str(os.getpid())],text=True,timeout=5)
        return int(value.strip())*1024
    signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        record['before_admission_rss_bytes'] = rss()
        spec = importlib.util.spec_from_file_location('exact_frozen_native_admission',HERE/'native.py')
        native = importlib.util.module_from_spec(spec); sys.modules[spec.name] = native
        spec.loader.exec_module(native)
        context = native.authenticate('terminal',native.BUILD_SHA,native.CANDIDATES['terminal'],
            native.BUILDER_SHA,NATIVE_SHA,WATCH_SHA,
            '459122d34f41759700e0cdff43c799072afe1be9853eccee3dc4e83e36680fc4',
            '8fd4a858bb281ec665c0ec32b179f17f67e798da51aec414e3498f9e05635a66')
        record['after_admission_rss_bytes'] = rss()
        record['gc_collected'] = gc.collect()
        record['after_full_gc_rss_bytes'] = rss()
        assert not any(x in sys.modules for x in ('numpy','torch','flopscope'))
        native.bind(context['source_bindings']); native.bind(context['input_bindings'])
        record.update(source_bindings=context['source_bindings'],input_bindings=context['input_bindings'],
            context_retained=True,estimator_candidate_sha256=native.CANDIDATES['terminal'],
            passed=True,completed=True,hashes_unchanged=True)
    except BaseException:
        record['error'] = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        record.update(elapsed_s=time.monotonic()-start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write(out/'result.json',record)
        write(out/'completion.json',dict(passed=record['passed'],completed=record['completed'],
                                        result_sha256=sha(out/'result.json')))
        print(json.dumps({k:v for k,v in record.items() if k not in ('source_bindings','input_bindings')},allow_nan=False))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
