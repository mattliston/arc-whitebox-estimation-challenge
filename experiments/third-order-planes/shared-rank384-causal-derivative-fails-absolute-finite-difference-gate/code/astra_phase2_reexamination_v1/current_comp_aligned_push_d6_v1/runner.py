"""Inert adapter for the fixed two-context aligned-PUSH runtime falsifier."""
import argparse
import hashlib
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OWNER = NEW / 'run_remote_series_v1.py'
OWNER_SHA256 = '00eabab0b7c1ec0c113b998032d441b60aa882670f364732f02a741282634cb1'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def source_and_proof():
    """Stdlib-only source derivation; does not launch or import either runtime."""
    original = OWNER.read_bytes()
    require(hashlib.sha256(original).hexdigest() == OWNER_SHA256,
            'Frozen original persistent-server runner owner')
    source = original.decode('utf-8')
    before = 'str(HERE/"check_remote_series_v1.py"),"--candidate"'
    after = ('str(HERE/"current_comp_aligned_push_d6_v1/client.py"),'
             '"--run","--candidate"')
    require(source.count(before) == 1, 'Unique original client argv span')
    changed = source.replace(before, after)
    require(changed.count(after) == 1 and changed.replace(after, before) == source,
            'Exact whole-source inverse: client path and explicit run flag only')
    return changed, dict(
        schema='current_comp_aligned_push_d6_runner_adapter_v1',
        owner_path=str(OWNER), owner_sha256=OWNER_SHA256,
        derived_source_sha256=hashlib.sha256(changed.encode('utf-8')).hexdigest(),
        whole_source_inverse_verified=True,
        only_client_path_and_run_flag_changed=True,
        client_path=str(HERE / 'client.py'), client_run_flag_forwarded=True,
        derived_file_path=str(OWNER), original_here_and_cwd_preserved=True,
        original_server_environment_and_resource_loop_unchanged=True,
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
    name = '_current_comp_aligned_push_d6_runner_body'
    require(name not in sys.modules, 'Fresh runner body; no repeated launch')
    runtime = types.ModuleType(name)
    runtime.__file__ = str(OWNER)
    runtime.__package__ = None
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
