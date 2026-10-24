"""DRAFT: original two-context CB2 client plus closed scalar W5 diagnostics.

No candidate/fixture/numerical import occurs before the explicit --run branch.
This file has not been executed or preflighted.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OWNER = NEW/'current_comp_compact_frames_v1/client.py'
OWNER_SHA = 'de87726e4b1feae01e2a36d586c6103e68910f0f7f316c586b75ccd7e4a30e0e'
SERIES = NEW/'check_remote_series_v1.py'
PINS = {
    OWNER: OWNER_SHA,
    NEW/'current_comp_padded_push_d6_v1/client.py':
        '46979737953192110b4c70f04d4ea9a8cd106ace217c1d9d4e8c8e6cb29e0de3',
    SERIES: 'f3b0b9dd9e9edd27bf4f85b5a34ab0d72131e22881dd0531b3201dc29642da55',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _authenticate():
    for path, digest in PINS.items():
        require(path.is_file() and not path.is_symlink()
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                'Exact complete original client owner')


DIAGNOSTICS = '''def _w5_closed_metadata(module):
    # Host scalar metadata only, after the original paid context has closed.
    value = module._W5_PORT_LAST
    assert module._W5_PORT_SCOPE is None and isinstance(value,dict)
    assert value['completed'] is True and value['numeric_cache_retired'] is True
    assert value['additional_initialization_flops'] == 0
    rows = value['layers']
    assert isinstance(rows,list) and len(rows) == 16
    creators = hits = copied = removed = capacity = 0
    clean = []
    for layer,row in enumerate(rows):
        assert isinstance(row,dict) and row['layer'] == layer
        k = row['previous_count']
        assert type(k) is int and 0 < k <= 1024 and k%32 == 0
        roles = {} if layer == 0 else dict(cov_outer=1,newborn_common=1,star_push=1)
        if layer >= 5:
            roles['shared_basis'] = 1
        assert row['roles'] == roles
        c,h = (0,0) if layer == 0 else (1,len(roles)-1)
        fcopy,fremoved = (0,0) if layer == 0 else (16807*k,h*36827*k)
        cap = 0 if layer == 0 else 67228*(1<<(k-1).bit_length())
        expected = dict(layer=layer,previous_count=k,roles=roles,creators=c,hits=h,
                        copy_flops=fcopy,removed_flops=fremoved,capacity_bytes=cap)
        assert row == expected
        assert all(type(row[key]) is int for key in
                   ('layer','previous_count','creators','hits','copy_flops',
                    'removed_flops','capacity_bytes'))
        clean.append(expected)
        creators += c; hits += h; copied += fcopy; removed += fremoved
        capacity = max(capacity,cap)
    assert (creators,hits,copied,removed,capacity) == (
        15,41,230188672,1363482848,68841472)
    expected = dict(completed=True,layers=clean,creators=creators,hits=hits,
                    copied_flops=copied,removed_flops=removed,
                    maximum_cache_bytes=capacity,numeric_cache_retired=True,
                    additional_initialization_flops=0)
    assert value == expected
    expected['net_saved_flops'] = removed-copied
    expected['source_certified_arithmetic_copy_calls_saved'] = 30*hits-creators
    return expected


'''


def source_and_proof():
    """Exact effective frame-client inverse; only scalar diagnostics are new."""
    _authenticate()
    spec = importlib.util.spec_from_file_location('_w5_full_client_owner',OWNER)
    owner = importlib.util.module_from_spec(spec); spec.loader.exec_module(owner)
    original, parent = owner.source_and_proof()
    edits = (
        ('def run_one(index,rank):\n',DIAGNOSTICS+'def run_one(index,rank):\n'),
        ('                compact_binding_events=mod._CS_BINDING_CALLS-compact_before[1])',
         '                compact_binding_events=mod._CS_BINDING_CALLS-compact_before[1],\n'
         '                weight_reuse=_w5_closed_metadata(mod))'),
    )
    source, positions = original, []
    for before, after in edits:
        require(source.count(before) == 1,'One exact closed scalar diagnostic splice')
        at = source.index(before); positions.append((at,before,after))
        source = source[:at]+after+source[at+len(before):]
    inverse = source
    for at,before,after in reversed(positions):
        require(inverse[at:at+len(after)] == after,'Exact positional client inverse')
        inverse = inverse[:at]+before+inverse[at+len(after):]
    require(inverse == original,'Entire qualified two-context client restored')
    return source,dict(parent=parent,owner_sha256=OWNER_SHA,
        source_bindings={str(path):digest for path,digest in PINS.items()},
        derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        full_source_inverse_verified=True,closed_scalar_diagnostics_only=True,
        original_context_fixture_output_arithmetic_and_repeat_unchanged=True,
        ranks=[37,37],model_calls=2,
        removed_primitive_count_is_source_certificate_not_total_RPC_count=True,
        numerical_work_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--candidate'); parser.add_argument('--tag')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    if not args.candidate or not args.tag:
        parser.error('--run requires --candidate and --tag')
    source,unused = source_and_proof()
    name = '_current_comp_weight_reuse_full_client'
    require(name not in sys.modules,'One fresh persistent two-context client')
    runtime = types.ModuleType(name)
    runtime.__file__ = str(SERIES); runtime.__package__ = None
    sys.modules[name] = runtime
    saved = sys.argv
    try:
        sys.argv = [str(SERIES),'--candidate',args.candidate,'--tag',args.tag]
        exec(compile(source,str(SERIES),'exec'),vars(runtime))
    finally:
        sys.argv = saved
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
