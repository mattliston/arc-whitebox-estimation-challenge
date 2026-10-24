"""Run an existing private worker, then its stdlib comparator in the same process.

This entry creates no subprocess. The worker retains responsibility for acquiring
the shared lane lock and asserting nice/thread/pressure requirements. Its imported
safety module and flock remain alive until this entire entry exits.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

ASTRA_ROOT = Path(__file__).resolve().parents[1]


def private_path(text):
    path = Path(text).resolve()
    path.relative_to(ASTRA_ROOT)
    return path


def run_script(path, args):
    old_argv, old_path = sys.argv, list(sys.path)
    sys.argv = [str(path), *args]
    sys.path.insert(0, str(path.parent))
    try:
        try:
            runpy.run_path(str(path), run_name='__main__')
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
    finally:
        sys.argv, sys.path[:] = old_argv, old_path


class Tee:
    def __init__(self, stream):
        self.stream, self.parts = stream, []

    def write(self, text):
        self.parts.append(text)
        return self.stream.write(text)

    def flush(self):
        return self.stream.flush()


class FileTee:
    """Preserve the worker's stream while recording Python stdout/stderr."""
    def __init__(self, stream, record):
        self.stream, self.record = stream, record

    def write(self, text):
        self.record.write(text)
        self.record.flush()
        return self.stream.write(text)

    def flush(self):
        self.record.flush()
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True, type=private_path)
    parser.add_argument('--run', required=True, type=private_path)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--comparator', required=True, type=private_path)
    parser.add_argument('--compare-arg', action='append', default=[])
    parser.add_argument('--artifact', required=True, type=private_path)
    args = parser.parse_args()
    log = args.artifact.with_suffix('.log')
    worker_log = args.artifact.with_suffix('.worker.log')
    assert not any(p.exists() for p in (args.artifact, log, worker_log)), 'Comparison artifacts must be fresh'
    # One interleaved file lets comparators associate carrier stderr telemetry
    # with the worker's stdout completion events. No worker output stays in RAM.
    with worker_log.open('x', buffering=1) as record:
        with contextlib.redirect_stdout(FileTee(sys.stdout, record)), contextlib.redirect_stderr(FileTee(sys.stderr, record)):
            run_script(args.worker, [str(args.run), str(args.index)])
    # A drained worker with unavailable truths or errors must not appear successful.
    assert not list((args.run / 'queue').glob('*.json')), 'Worker left queued requests'
    assert not list((args.run / 'results').glob('*.error')), 'Worker published an error'
    rec = dict(worker=str(args.worker), run=str(args.run), comparator=str(args.comparator),
               compare_args=args.compare_arg, worker_log=str(worker_log), pid=os.getpid(),
               completed=False, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    tee = Tee(sys.stdout)
    try:
        with contextlib.redirect_stdout(tee):
            run_script(args.comparator, args.compare_arg)
        rec['completed'] = True
    except BaseException:
        rec['error'] = traceback.format_exc()
        raise
    finally:
        output = ''.join(tee.parts)
        rec['stdout'] = output
        # The comparator prints a prose parity line followed by one JSON table.
        # Preserve the raw stdout regardless, and expose the table when present.
        for i, char in enumerate(output):
            if char == '{':
                try:
                    obj, end = json.JSONDecoder().raw_decode(output[i:])
                    if not output[i + end:].strip():
                        rec['comparison'] = obj
                        break
                except json.JSONDecodeError:
                    pass
        with args.artifact.open('x') as stream:
            json.dump(rec, stream, indent=2, allow_nan=False)
            stream.write('\n')
        with log.open('x') as stream:
            stream.write(output)


if __name__ == '__main__':
    main()
