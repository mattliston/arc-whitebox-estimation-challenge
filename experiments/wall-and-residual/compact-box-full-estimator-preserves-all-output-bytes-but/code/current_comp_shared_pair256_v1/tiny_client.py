"""Source-hashed NumPy-free entry to the fixed pair256 local-IPC client mode."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--source-sha256')
    parser.add_argument('--client-sha256')
    args,_ = parser.parse_known_args()
    if not args.run:
        print('Inert pair256 client. Only the reviewed watch may select --run --mode remote.')
        return 0
    worker = HERE/'tiny_worker.py'
    if (hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != args.client_sha256
            or hashlib.sha256(worker.read_bytes()).hexdigest() != args.source_sha256
            or importlib.util.find_spec('numpy') is not None
            or any(k.split('.')[0] in ('numpy','scipy','torch','flopscope') for k in sys.modules)):
        raise RuntimeError('Exact client/worker owners and genuinely NumPy-free fresh process')
    spec = importlib.util.spec_from_file_location('pair256_tiny_client_worker_owner',worker)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(forced_mode='remote')


if __name__ == '__main__':
    raise SystemExit(main())
