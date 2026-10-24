"""Inert wrapper: the original NumPy-free five-call client body is byte-exact."""
import argparse
import hashlib
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent/'check_remote_series_v1.py'
OWNER_SHA256 = 'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55'


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def source_and_proof():
    raw = OWNER.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == OWNER_SHA256,'Original NumPy-free production client')
    source = raw.decode('utf-8')
    require(source.encode() == raw and source.count('enumerate((35,100,500,3016500,35))') == 1,
        'Byte-exact original numerical body and fixed five-call roster')
    return source,dict(owner_sha256=OWNER_SHA256,derived_source_sha256=OWNER_SHA256,
        whole_source_inverse_verified=True,original_client_body_byte_exact=True,
        ranks=[35,100,500,3016500,35],full_model_calls=5,persistent_estimator=True,
        original_here_and_fixture_loading_preserved=True,original_context_and_output_arithmetic_unchanged=True,
        original_repeat_output_hash_check_unchanged=True,estimator_source_edited=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--candidate'); parser.add_argument('--tag')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    if not args.candidate or not args.tag:
        parser.error('--run requires --candidate and --tag')
    source,_ = source_and_proof()
    name = '_current_comp_shared_old384_remote_client'
    require(name not in sys.modules,'One fresh persistent client body')
    runtime = types.ModuleType(name)
    runtime.__file__ = str(OWNER); runtime.__package__ = None
    sys.modules[name] = runtime  # Required by the unchanged @dataclass owner.
    saved = sys.argv
    try:
        sys.argv = [str(OWNER),'--candidate',args.candidate,'--tag',args.tag]
        exec(compile(source,str(OWNER),'exec'),vars(runtime))
    finally:
        sys.argv = saved
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
