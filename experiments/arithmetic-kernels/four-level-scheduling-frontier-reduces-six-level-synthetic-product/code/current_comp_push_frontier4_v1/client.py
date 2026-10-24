"""DRAFT: NumPy-free local-IPC entry for the fixed frontier4 comparison."""
import hashlib
import importlib.util
from pathlib import Path
import sys


def main():
    if '--run' not in sys.argv:
        print('Inert RPC kernel client; explicit --run and reviewed SHA arguments required.')
        return 0
    def flag(name):
        if sys.argv.count(name)!=1:
            raise RuntimeError('Exactly one explicit source authority '+name)
        return sys.argv[sys.argv.index(name)+1]
    here=Path(__file__).resolve().parent;worker=here/'worker.py'
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=flag('--client-sha256'):
        raise RuntimeError('Exact current client source')
    if hashlib.sha256(worker.read_bytes()).hexdigest()!=flag('--source-sha256'):
        raise RuntimeError('Exact worker before delegating the remote entry')
    if importlib.util.find_spec('numpy') is not None:
        raise RuntimeError('Actual client must have no NumPy installation')
    spec=importlib.util.spec_from_file_location('micro3_actual_remote_worker',worker)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.main(forced_mode='remote')


if __name__=='__main__':
    raise SystemExit(main())
