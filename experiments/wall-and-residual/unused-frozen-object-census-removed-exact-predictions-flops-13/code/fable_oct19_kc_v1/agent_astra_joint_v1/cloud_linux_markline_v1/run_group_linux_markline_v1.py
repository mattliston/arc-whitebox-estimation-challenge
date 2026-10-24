"""One owned client/server pair. Must be a child of guard_linux_v1.py."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from common_linux_v7 import THREADS, thread_provenance, candidate_path, load_plan, sha, utc, write_new


def stop(p):
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(timeout=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--candidate', required=True); p.add_argument('--ranks', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--client-python', type=Path, required=True); p.add_argument('--server-python', type=Path, required=True)
    p.add_argument('--setup', choices=['paced', 'real'], default='paced'); p.add_argument('--warmup-each', action='store_true')
    a = p.parse_args()
    threads = thread_provenance()
    assert sys.platform == 'linux'
    assert os.environ.get('ARC_GUARDED_PGID') == str(os.getpgrp()), 'must be launched by Linux guard'
    a.out.mkdir(parents=True, exist_ok=False)
    plan = load_plan(a.root, a.plan); rec = plan['candidates'][a.candidate]
    candidate = candidate_path(a.root, rec, a.out)
    here = Path(__file__).resolve().parent
    server = client = None; exit_code = 1; error = None
    env = dict(os.environ, **THREADS, ARC_BUNDLE_ROOT=str(a.root.resolve()), ARC_RUN_PLAN=str(a.plan.resolve()),
               FLOPSCOPE_MAX_ARRAY_COUNT='100000', FLOPSCOPE_MAX_ARRAY_BYTES='4294967296',
               RELEASE_SERVER_RECEIPT=str(a.out.resolve() / 'server.json'),
               RELEASE_SERVER_MARKS=str(a.out.resolve() / 'server_marks.jsonl'))
    def interrupted(*_):
        raise KeyboardInterrupt('guard signal')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    with tempfile.TemporaryDirectory(prefix='arc-') as td, (a.out / 'server.log').open('x') as server_log:
        socket = Path(td) / 's'; url = 'ipc://' + str(socket)
        try:
            server = subprocess.Popen([str(a.server_python), '-B', '/home/arcworker/arc-workspace/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1/cloud_linux_capacity_v7/server_linux_v7.py', '--url', url, '--timeout', '1800'],
                                      env=env, stdout=server_log, stderr=subprocess.STDOUT)
            for _ in range(600):
                if server.poll() is not None:
                    raise RuntimeError('server exited before startup')
                if socket.exists():
                    break
                time.sleep(.05)
            else:
                raise RuntimeError('server did not create socket within30s')
            command = [str(a.client_python), '-B', '-u', str(here / 'client_linux_markline_v1.py'), '--root', str(a.root.resolve()),
                       '--plan', str(a.plan.resolve()), '--candidate', str(candidate.resolve()),
                       '--candidate-sha', rec['effective_sha256'], '--ranks', a.ranks, '--out', str(a.out.resolve()), '--setup', a.setup]
            if a.warmup_each:
                command.append('--warmup-each')
            client = subprocess.Popen(command, env=dict(env, FLOPSCOPE_SERVER_URL=url, RELEASE_SERVER_PID=str(server.pid)))
            exit_code = client.wait()
        except BaseException as exc:
            error = repr(exc)
        finally:
            stop(client); stop(server)
            write_new(a.out / 'launcher.json', dict(utc=utc(), candidate=a.candidate, source_sha256=rec['effective_sha256'],
                      plan_sha256=sha(a.plan), group=os.getpgrp(), client_exit=client.returncode if client else None,
                      server_exit=server.returncode if server else None, error=error, handle_cap=100000,
                      array_bytes_cap=4294967296, setup_mode=a.setup, warmup_each=a.warmup_each,
                      science_only=True, release_gate=False, thread_provenance=threads))
    return exit_code if error is None else 1


if __name__ == '__main__':
    sys.exit(main())
