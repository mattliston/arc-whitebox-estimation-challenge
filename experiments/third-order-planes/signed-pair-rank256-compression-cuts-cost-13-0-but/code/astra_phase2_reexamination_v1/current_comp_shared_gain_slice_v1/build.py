"""Inert one-span gain-slice builder. No estimator, numerical or target imports."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / 'current_comp_shared_old384_v1'
PARENT = SHARED / 'build_v1/estimator.py'
PARENT_SHA = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
OUTPUT = HERE / 'build_v1'
SCHEMA = 'current_comp_shared_gain_slice_build_v1'
SOURCE_PINS = {
    PARENT: PARENT_SHA,
    SHARED / 'scalar_semantics_probe_v3.py': 'e6336483af82f75364efe64a95eaa0855a676c19ceaf878e5d74d9908580afca',
    SHARED / 'SCALAR_SEMANTICS_PROTOCOL_V3.md': '9ff60bc1ab796df0e9911119a0f0bc24a1ee063226a21f6c6bea0e5dec489f6f',
    SHARED / 'watch_scalar_semantics_v3.py': '42f643f2d484f39faa52060d36f7983b13c59b083f764df36899245f4a26f297',
    SHARED / 'SCALAR_SEMANTICS_WATCH_V3_SPEC.md': '691da41946775190b1652f40f047dcaffc267c6a823328d390a59b4832171c12',
}
SCALAR_SEALS = {
    'scalar_semantics_v3/local': (
        'e1025386a0a6ab5be11630ed59b9b1abeef35b53b0d7b84667c55fc34a62d9c0',
        '7791c0631e9e8e049a74f6700642ab15f8d8805eb4dd5cee7f0010bbf76c4101',
        'b5a91fcc95cdf93bed70908cf4ec46f3f7f5248626886f9ab4e2bb40b67f6f6a'),
    'scalar_semantics_v3/remote': (
        '9bf16e7ebf2f8a6373f86ca8526340e94418a2ef1ee164613874829ab29d0d13',
        '82c8bf07648beab195a4fa9a97a65b6b5d9ca874664b07e73692cf261f1fd194',
        '742428c039b8c818fd776803223e1cc1cde24e9bdeb34fc7186243d081db8a6a'),
    'scalar_semantics_watch_v3': (
        '31e484037eb7a4cb2965be9ebb16d5a8ee749c305cc240876b48c71e8d672541',
        'fc1ca2aaeb5e8a09a2621469f3a4194cb6e538f0d8843ee154ae04fc4e81389d',
        '73a76214e4a0a5ecf35e68c28962656e05f4417bfdfdf1f3bd4d11e969c1f1b4'),
}
SEAL_NAMES = ('result.json', 'completion.json', 'finalization.json')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path, tick=None):
    path = Path(path)
    require(not path.is_symlink() and 'truthpost' not in path.parts, 'Regular target-free owner')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
            if tick:
                tick()
    return h.hexdigest()


def read(path):
    require(Path(path).stat().st_size < 16 * 1024**2, 'Bounded metadata')
    return json.loads(Path(path).read_text())


def bind(mapping, tick=None):
    for path, digest in mapping.items():
        require(sha(path, tick) == digest, 'Exact source/input: ' + str(path))


def authenticate():
    bind(SOURCE_PINS)
    return {str(p): d for p, d in SOURCE_PINS.items()}


def scalar_prerequisite(tick=None):
    inputs, records = {}, {}
    for folder, expected in SCALAR_SEALS.items():
        paths = [SHARED / folder / n for n in SEAL_NAMES]
        ds = tuple(sha(p, tick) for p in paths)
        require(ds == expected, 'Actual scalar-v3 nine-seal evidence')
        r, e, f = map(read, paths)
        require(all(x.get('passed') is True and x.get('completed') is True for x in (r, e, f))
                and e['result_sha256'] == f['result_sha256'] == ds[0]
                and f['completion_sha256'] == ds[1] and f['elapsed_s'] < 60,
                'Positive linked scalar evidence, never old failed harness')
        records[folder] = r
        inputs.update({str(p): d for p, d in zip(paths, ds)})
    w = records['scalar_semantics_watch_v3']
    require(w['scalar_hypothesis_supported'] and w['model_calls'] == 0 and w['truth_accessed'] is False
            and w['comparison']['measured_local_minus_remote_complete_flops'] == 15090
            and w['comparison']['slice_control_all_intermediate_bytes_exact']
            and all(w[k] for k in ('whole_owned_tree_retired', 'cleanup_passed', 'observational_groups_retired'))
            and w['maximum_combined_rss_bytes'] < 2_000_000_000 and w['maximum_interval_s'] <= 1
            and w['maximum_swap_growth_bytes'] == 0, 'Actual target-free interface evidence only')
    for mode, expected in (('local', 94780), ('remote', 79690)):
        r = records['scalar_semantics_v3/' + mode]
        require(r['mode'] == mode and r['model_calls'] == 0 and r['truth_accessed'] is False
                and r['measurement']['complete_flops'] == expected
                and r['proof']['candidate_sha256'] == PARENT_SHA, 'Same parent and complete actual tiny bill')
    for mapping in (w['source_bindings'], w['input_bindings'], w['artifacts']):
        bind(mapping, tick)
        for path, digest in mapping.items():
            require(path not in inputs or inputs[path] == digest, 'Consistent direct scalar inventory')
            inputs[path] = digest
    return inputs


def derive_source():
    raw = PARENT.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PARENT_SHA, 'Immutable SHARED parent')
    original = raw.decode('utf-8')
    before, after = 'fnp.tanh(P["gain"][l])', 'fnp.tanh(P["gain"][l:l+1])'
    require(original.count(before) == 1 and after not in original, 'The one active gain tanh site')
    changed = original.replace(before, after)
    require(changed.count(after) == 1 and changed.replace(after, before).encode() == raw,
            'Whole-source inverse preserves every other byte and literal')
    ast.parse(changed)
    proof = dict(parent_sha256=PARENT_SHA, candidate_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                 one_span_gain_slice=True, before=before, after=after, whole_source_inverse_verified=True,
                 all_learned_literals_byte_exact=True, no_new_parameter_or_head=True,
                 shared_basis_and_physics_source_unchanged=True, candidate_imported=False,
                 local_full_network_equality_not_assumed=True, full_fidelity_not_yet_qualified=True)
    return changed, proof


def verify_build(build_sha, candidate_sha, builder_sha, protocol_sha, tick=None):
    require(sha(HERE / 'build.py', tick) == builder_sha and sha(HERE / 'BUILD_PROTOCOL.md', tick) == protocol_sha,
            'Explicit current builder/protocol')
    paths = [OUTPUT / n for n in SEAL_NAMES]
    ds = [sha(p, tick) for p in paths]
    r, e, f = map(read, paths)
    require(ds[0] == build_sha and all(x.get('passed') is True and x.get('completed') is True for x in (r, e, f))
            and e['result_sha256'] == f['result_sha256'] == ds[0] and f['completion_sha256'] == ds[1]
            and r['schema'] == SCHEMA and r['candidate_sha256'] == candidate_sha
            and r['estimator_executed'] is False and r['truth_accessed'] is False and r['parameter_updates'] == 0
            and 0 <= r['elapsed_s'] < 60 and 0 <= f['elapsed_s'] < 60 and r['peak_rss_bytes'] < 2_000_000_000,
            'Actual source-only build three seals')
    text, proof = derive_source()
    require(proof == r['proof'] and proof['candidate_sha256'] == candidate_sha
            and (OUTPUT / 'estimator.py').read_bytes() == text.encode(), 'Actual sole built one-span candidate')
    for group in ('source_bindings', 'input_bindings', 'artifacts'):
        bind(r[group], tick)
    require(r['source_bindings'][str(HERE / 'build.py')] == builder_sha
            and r['source_bindings'][str(HERE / 'BUILD_PROTOCOL.md')] == protocol_sha, 'Matching build authorities')
    return dict(build=r, candidate=str(OUTPUT / 'estimator.py'), candidate_sha256=candidate_sha, proof=proof,
                source_bindings=r['source_bindings'], input_bindings={**r['input_bindings'], **r['artifacts'],
                **{str(p): d for p, d in zip(paths, ds)}})


def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source-sha256'); parser.add_argument('--protocol-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    began = time.monotonic()
    require(not any(n.split('.')[0] in ('numpy', 'flopscope', 'torch', 'scipy') for n in sys.modules),
            'Source-only build before numerical imports')
    OUTPUT.mkdir(exist_ok=False)
    r = dict(schema=SCHEMA, passed=False, completed=False, estimator_executed=False, truth_accessed=False,
             parameter_updates=0, source_bindings={}, input_bindings={}, artifacts={})
    def tick():
        require(time.monotonic() - began < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000,
                'Whole source build60s/self2GB; external supervisor owns tree')
    try:
        sources = {**authenticate(), str(HERE / 'build.py'): args.source_sha256,
                   str(HERE / 'BUILD_PROTOCOL.md'): args.protocol_sha256}
        bind(sources, tick)
        r.update(source_bindings=sources, input_bindings=scalar_prerequisite(tick))
        text, proof = derive_source(); tick()
        with (OUTPUT / 'estimator.py').open('xb') as stream:
            stream.write(text.encode())
        r.update(candidate=str(OUTPUT / 'estimator.py'), candidate_sha256=proof['candidate_sha256'], proof=proof)
        r['artifacts'][str(OUTPUT / 'estimator.py')] = sha(OUTPUT / 'estimator.py', tick)
        r['artifacts'][str(OUTPUT / 'proof.json')] = write(OUTPUT / 'proof.json', proof)
        for group in ('source_bindings', 'input_bindings', 'artifacts'):
            bind(r[group], tick)
        require(not any(n.split('.')[0] in ('numpy', 'flopscope', 'torch', 'scipy') for n in sys.modules),
                'No numerical import through successful build')
        r.update(passed=True, completed=True, sources_unchanged=True, hashes_unchanged=True)
    except BaseException as error:
        r.update(passed=False, completed=False, error_type=type(error).__name__, error=str(error)[:512])
    r.update(elapsed_s=time.monotonic() - began, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if r['elapsed_s'] >= 60 or r['peak_rss_bytes'] >= 2_000_000_000:
        r.update(passed=False, completed=False)
    rh = write(OUTPUT / 'result.json', r)
    ok = bool(r['passed'] and time.monotonic() - began < 60)
    eh = write(OUTPUT / 'completion.json', dict(passed=ok, completed=ok, result_sha256=rh))
    elapsed = time.monotonic() - began; ok = bool(ok and elapsed < 60)
    write(OUTPUT / 'finalization.json', dict(passed=ok, completed=ok, result_sha256=rh,
                                           completion_sha256=eh, elapsed_s=elapsed))
    ok = bool(ok and time.monotonic() - began < 60)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
