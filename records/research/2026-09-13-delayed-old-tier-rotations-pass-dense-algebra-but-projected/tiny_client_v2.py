"""DRAFT: exact frozen NumPy-free client with the V2 worker filename only."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE/'tiny_client.py'
OWNER_SHA = '2e00d4fd9cb52cbb4fe8839d8c3165fce6205f58e65048c40a262ec0605415b7'


def derive_source():
    if OWNER.is_symlink() or hashlib.sha256(OWNER.read_bytes()).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact frozen V1 NumPy-free client adapter')
    spec = importlib.util.spec_from_file_location('_lazy_v1_client_source',OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.derive_source()
    before,after = b"HERE/'tiny_worker.py'",b"HERE/'tiny_worker_v2.py'"
    if original.count(before)!=1 or after in original:
        raise RuntimeError('One actual V2 client worker path')
    changed = original.replace(before,after)
    if changed.replace(after,before)!=original:
        raise RuntimeError('Complete V1 effective client inverse')
    return changed


def main():
    namespace = dict(__file__=str(Path(__file__).resolve()),__name__='_lazy_v2_client_effective')
    exec(compile(derive_source(),str(Path(__file__).resolve())+':V1-client-inverse','exec'),namespace)
    return namespace['main']()


if __name__=='__main__':
    raise SystemExit(main())
