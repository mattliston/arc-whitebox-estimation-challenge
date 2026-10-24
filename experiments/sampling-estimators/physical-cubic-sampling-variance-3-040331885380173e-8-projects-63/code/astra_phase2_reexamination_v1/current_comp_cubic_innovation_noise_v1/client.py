"""Exact NumPy-free qualified IPC client entry, only the worker path changes."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_shared_pair256_v1/tiny_client.py'
OWNER_SHA='a8876ee4dffab18e5c468c90a91ef7c96289264ecf5014d95b2feea7cdf6ca10'


def derive_source():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact qualified NumPy-free client source')
    original=raw.decode()
    before,after="HERE/'tiny_worker.py'","HERE/'worker.py'"
    if original.count(before)!=1 or after in original:
        raise RuntimeError('One exact new worker-path substitution')
    source=original.replace(before,after)
    if source.replace(after,before)!=original:
        raise RuntimeError('Entire qualified client restored')
    return source,dict(owner_sha256=OWNER_SHA,whole_client_inverse=True,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest())


def main():
    source,proof=derive_source()
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_noise_client_inert_owner')
    exec(compile(source,str(Path(__file__).resolve())+':whole-client-inverse','exec'),env)
    return env['main']()


if __name__=='__main__':
    raise SystemExit(main())
