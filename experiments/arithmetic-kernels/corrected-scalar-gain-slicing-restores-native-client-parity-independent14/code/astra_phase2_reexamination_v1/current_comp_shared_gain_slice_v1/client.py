"""Inert fixed35/35 adapter of the original NumPy-free persistent client."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OWNER = NEW / 'check_remote_series_v1.py'
OWNER_SHA256 = 'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55'
TAG = 'current_comp_shared_gain_slice_pair_v1'
CANDIDATE = 'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
STORE = NEW.parent / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
ROW = STORE / 'rows/observer_fit/0035'
PINS = {OWNER: OWNER_SHA256,
    STORE / 'manifest.json': '9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50',
    ROW / 'row_manifest.json': '5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809',
    ROW / 'weights.npy': '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    path = Path(path)
    require(not path.is_symlink() and path.resolve().is_relative_to(NEW.parent)
            and 'truthpost' not in path.parts, 'Exact regular target-free input')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_and_proof():
    raw = OWNER.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == OWNER_SHA256, 'Original production client owner')
    original = raw.decode('utf-8')
    before, after = 'enumerate((35,100,500,3016500,35))', 'enumerate((35,35))'
    require(original.count(before) == 1 and after not in original, 'One fixed roster splice')
    source = original.replace(before, after)
    require(source.count(after) == 1 and source.replace(after, before).encode() == raw,
            'Full original client inverse; paid arithmetic and environment unchanged')
    return source, dict(owner=str(OWNER), owner_sha256=OWNER_SHA256,
        derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        whole_source_inverse_verified=True, source_splices=1, ranks=[35,35], full_model_calls=2,
        persistent_estimator=True, original_here_and_fixture_loading_preserved=True,
        original_context_and_output_arithmetic_unchanged=True,
        original_repeat_output_hash_check_unchanged=True, no_runner_or_inner_monitor=True,
        input_preparation_convention='original fnp.load inside participant context; actual complete bill')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--candidate'); parser.add_argument('--tag')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    require(args.candidate == CANDIDATE and args.tag == TAG, 'Only the selected fresh gain-slice35 pair')
    require(not any(n.split('.')[0] in ('numpy','scipy','torch','flopscope','threadpoolctl')
                    for n in sys.modules), 'Metadata admission before any numerical package')
    require(not any(p.exists() for p in (NEW / (TAG + '.json'),
                *(NEW / f'{TAG}_{i}_r35.f32' for i in range(2)))), 'Exclusive original client outputs')
    for path, digest in PINS.items():
        require(sha(path) == digest, 'Canonical source/physical35 input')
    paths = [HERE / 'build_v1' / n for n in ('result.json','completion.json','finalization.json')]
    records = []
    for path in paths:
        require(not path.is_symlink() and path.stat().st_size < 16 * 1024**2, 'Bounded actual build metadata')
        records.append(json.loads(path.read_text()))
    build, end, final = records
    require(all(x.get('passed') is True and x.get('completed') is True for x in records)
            and end['result_sha256'] == final['result_sha256'] == sha(paths[0])
            and final['completion_sha256'] == sha(paths[1])
            and build['schema'] == 'current_comp_shared_gain_slice_build_v1'
            and build['candidate'] == str(NEW / CANDIDATE)
            and build['candidate_sha256'] == sha(NEW / CANDIDATE), 'Actual built candidate, never a hypothetical source')
    source, _ = source_and_proof()
    name = '_current_comp_shared_gain_slice_remote_pair'
    require(name not in sys.modules, 'One fresh persistent client body')
    runtime = types.ModuleType(name)
    runtime.__file__ = str(OWNER); runtime.__package__ = None
    sys.modules[name] = runtime  # Required by the original dataclass definition.
    saved = sys.argv
    try:
        sys.argv = [str(OWNER), '--candidate', args.candidate, '--tag', args.tag]
        exec(compile(source, str(OWNER) + ':fixed-gain-slice-pair', 'exec'), vars(runtime))
    finally:
        sys.argv = saved
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
