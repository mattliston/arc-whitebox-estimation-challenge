"""Inert adapter: exactly two original remote contexts, TRAIN37 then TRAIN37."""
import argparse
import hashlib
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OWNER = NEW / 'check_remote_series_v1.py'
OWNER_SHA256 = 'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def source_and_proof():
    """Stdlib-only roster inverse; no FNP, candidate, fixture or target import."""
    original = OWNER.read_bytes()
    require(hashlib.sha256(original).hexdigest() == OWNER_SHA256,
            'Frozen original NumPy-free persistent client owner')
    source = original.decode('utf-8')
    before = '    for index,rank in enumerate((35,100,500,3016500,35)):\n'
    after = '    for index,rank in enumerate((37,37)):\n'
    require(source.count(before) == 1, 'Unique original fixed context roster')
    changed = source.replace(before, after)
    require(changed.count(after) == 1 and changed.replace(after, before) == source,
            'Exact whole-source inverse: fixed two-context roster only')
    return changed, dict(
        schema='current_comp_padded_push_d6_client_adapter_v1',
        owner_path=str(OWNER), owner_sha256=OWNER_SHA256,
        derived_source_sha256=hashlib.sha256(changed.encode('utf-8')).hexdigest(),
        whole_source_inverse_verified=True, only_fixed_roster_changed=True,
        ranks=[37, 37], full_model_calls=2, persistent_estimator=True,
        derived_file_path=str(OWNER), original_here_and_fixture_loading_preserved=True,
        original_context_and_output_arithmetic_unchanged=True,
        original_repeat_output_hash_check_unchanged=True,
        estimator_source_edited=False, numerical_work_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--candidate')
    parser.add_argument('--tag')
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 0
    if not args.candidate or not args.tag:
        parser.error('--run requires --candidate and --tag')
    source, _ = source_and_proof()
    name = '_current_comp_padded_push_d6_client_body'
    require(name not in sys.modules, 'Fresh persistent client; no second series')
    runtime = types.ModuleType(name)
    runtime.__file__ = str(OWNER)
    runtime.__package__ = None
    # Register the actual derived namespace for the unchanged @dataclass owner.
    sys.modules[name] = runtime
    saved = sys.argv
    try:
        sys.argv = [str(OWNER), '--candidate', args.candidate, '--tag', args.tag]
        exec(compile(source, str(OWNER), 'exec'), vars(runtime))
    finally:
        sys.argv = saved
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
