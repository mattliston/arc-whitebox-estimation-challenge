"""DRAFT inert NumPy-free client, exact qualified entry with a descriptive name change."""
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'current_comp_shared_pair256_v1/tiny_client.py'
OWNER_SHA = 'a8876ee4dffab18e5c468c90a91ef7c96289264ecf5014d95b2feea7cdf6ca10'


def derive_source():
    original = OWNER.read_bytes()
    if hashlib.sha256(original).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact admitted NumPy-free client entry')
    changed = original.replace(b'pair256',b'lazy_rotation')
    if changed.replace(b'lazy_rotation',b'pair256')!=original:
        raise RuntimeError('Whole qualified client source inverse')
    return changed


def main():
    namespace = dict(__file__=str(Path(__file__).resolve()),__name__='lazy_rotation_client_inert')
    exec(compile(derive_source(),str(Path(__file__).resolve())+':source-inverse','exec'),namespace)
    return namespace['main']()


if __name__=='__main__':
    raise SystemExit(main())
