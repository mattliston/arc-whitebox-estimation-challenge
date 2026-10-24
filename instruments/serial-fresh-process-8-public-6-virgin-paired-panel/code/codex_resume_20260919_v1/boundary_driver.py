"""Own one server/client pair, inherit the external guarded process group."""
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
P2 = HERE.parent


def main():
    candidate, out, ranks = sys.argv[1:]
    destination = Path(out)
    assert not destination.exists()
    socket = Path(f'/tmp/cdr19-{os.getpid()}.sock')
    assert not socket.exists()
    url = 'ipc://' + str(socket)
    backend = P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
    client = P2/'data_gate/.client-venv/bin/python'
    with (HERE/(destination.name+'_server.log')).open('x') as log:
        server = subprocess.Popen([str(backend), '-m', 'flopscope_server', '--url', url, '--timeout', '600'], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+10
            while not socket.exists():
                assert server.poll() is None and time.monotonic()<deadline, 'Server startup failed'
                time.sleep(.05)
            env = dict(os.environ, FLOPSCOPE_SERVER_URL=url)
            done = subprocess.run([str(client), '-B', '-u', str(HERE/'boundary_client.py'),
                                   candidate, out, ranks], env=env, timeout=550)
            return done.returncode
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            socket.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
