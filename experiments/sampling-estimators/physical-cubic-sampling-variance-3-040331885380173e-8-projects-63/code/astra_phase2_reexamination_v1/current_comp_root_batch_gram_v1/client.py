"""Inert exact qualified NumPy-free entry; only the worker filename changes."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_shared_pair256_v1/tiny_client.py'
OWNER_SHA='a8876ee4dffab18e5c468c90a91ef7c96289264ecf5014d95b2feea7cdf6ca10'


def derive_source():
    original=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(original).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact admitted NumPy-free client owner')
    before,after=b"HERE/'tiny_worker.py'",b"HERE/'worker.py'"
    if original.count(before)!=1 or after in original:
        raise RuntimeError('One exact client worker path')
    source=original.replace(before,after)
    if source.replace(after,before)!=original:
        raise RuntimeError('Whole qualified NumPy-free client inverse')
    return source


def main():
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_root_batch_client_inert')
    exec(compile(derive_source(),str(Path(__file__).resolve())+':whole-client-inverse','exec'),env)
    return env['main']()


if __name__=='__main__':
    raise SystemExit(main())
