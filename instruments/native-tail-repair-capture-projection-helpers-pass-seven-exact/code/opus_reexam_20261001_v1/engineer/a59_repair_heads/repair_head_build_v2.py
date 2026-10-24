"""Strict stdlib-only T48 HU converter v2 (linked provenance) and native post-stack constructor.

No estimator or numerical library is imported. HU31/32 never enter the legacy
BAND converter. split_hu exposes only a 31-row incumbent table plus explicit
HC/HCF tables. construct verifies those weights against the actual native
parent before attaching the two new columns. Disabled/zero output is identical
to parent bytes. CLI: repair_head_build_v1.py PARENT SIDECAR NEW_OUTPUT.py
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import struct

HERE = Path(__file__).resolve().parent
SPEC_SHA = '29fd46fa9ad6243249bb4b4606f20e3cd78f6b282781061fda500a8b6a7af821'
SCHEMA = 'separate_repair_heads_v1'
BASE_ROWS = {6: 'diag', 20: 'k4_local_rows', 21: 'm2', 28: 'eold', 29: 'qb'}
FINAL_ROWS = {17: 'f15_diag'}
FLAGS = {'hsplit': {}, 'hbands': False, 'b7': False}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_string(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'invalid SHA256')
    return value


def pinned(pin):
    require(type(pin) is dict and set(pin) == {'path', 'sha256'}, 'pin must be path+sha256')
    path = Path(pin['path'])
    require(path.name != 'key.txt', 'credential files are forbidden')
    require(path.is_absolute() and path.is_file(), 'pin path must be an existing absolute file')
    expected = sha_string(pin['sha256'])
    data = path.read_bytes()
    require(sha(data) == expected, 'pinned file changed: ' + str(path))
    return data


def f32(value):
    require(type(value) in (int, float) and math.isfinite(value), 'finite numeric weight required')
    try:
        answer = struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error):
        raise ValueError('weight outside float32')
    require(math.isfinite(answer), 'nonfinite float32 weight')
    return answer


def table(value, n):
    require(type(value) is list and len(value) == n, 'wrong HU row count')
    require(all(type(row) is list and len(row) == 128 for row in value), 'wrong HU column count')
    return [[f32(v) for v in row] for row in value]


def packed(rows):
    # Canonical shape is fixed by schema: row-major little-endian float32 bytes.
    return b''.join(struct.pack('<f', v) for row in rows for v in row)


def split_hu(value):
    full = table(value, 42)
    allowed = set(BASE_ROWS) | set(FINAL_ROWS) | {31, 32}
    require(all(i in allowed or not any(row) for i, row in enumerate(full)),
            'unsupported/dormant HU column; BAND, B7 and old-k3 are forbidden here')
    incumbent = [row[:] for row in full]
    incumbent[31], incumbent[32] = [0.0] * 128, [0.0] * 128
    active = [i for i in BASE_ROWS if any(incumbent[i])]
    final = [i for i in FINAL_ROWS if any(incumbent[i])]
    return {'full': full, 'incumbent': incumbent,
            'incumbent_hu31': incumbent[:31],
            'hu_f32_sha256': sha(packed(full)),
            'incumbent_hu_f32_sha256': sha(packed(incumbent)),
            'u': [full[31], full[32]],
            'hc': {'columns': [BASE_ROWS[i] for i in active], 'u': [incumbent[i] for i in active], 'rows': active},
            'hcf': {'columns': [FINAL_ROWS[i] for i in final], 'u': [incumbent[i] for i in final], 'rows': final}}


def normalize(values):
    required = {'schema', 'enabled', 'feature_spec', 'parent_native_sha256', 'hu',
                'hu_f32_sha256', 'incumbent_hu_f32_sha256', 'u', 'required_flags', 'source'}
    require(type(values) is dict and set(values) == required, 'sidecar fields differ from schema')
    require(values['schema'] == SCHEMA and type(values['enabled']) is bool, 'schema/enabled invalid')
    require(values['feature_spec']['sha256'] == SPEC_SHA, 'unknown feature specification')
    spec = json.loads(pinned(values['feature_spec']))
    require(spec['rows'] == [31, 32] and spec['active_layers'] == list(range(7, 15)), 'feature spec changed')
    require(values['required_flags'] == FLAGS, 'HSPLIT/HBANDS/B7 incompatible with repair HU rows')
    full = split_hu(json.loads(pinned(values['hu'])))
    require(full['hu_f32_sha256'] == sha_string(values['hu_f32_sha256']), 'full HU canonical hash mismatch')
    require(full['incumbent_hu_f32_sha256'] == sha_string(values['incumbent_hu_f32_sha256']), 'incumbent HU canonical hash mismatch')
    require(packed(table(values['u'], 2)) == packed(full['u']), 'explicit repair weights differ from HU31/32')
    sha_string(values['parent_native_sha256'])
    source = values['source']
    require(type(source) is dict and set(source) == {'request', 'checkpoint', 'profile', 'dependencies', 'parent_binding'}, 'provenance fields invalid')
    request = json.loads(pinned(source['request']))
    profile = json.loads(pinned(source['profile']))
    pinned(source['checkpoint'])
    require(type(request.get('params')) is str and Path(request['params']).is_absolute()
            and Path(request['params']).resolve() == Path(source['checkpoint']['path']).resolve(), 'request/checkpoint path mismatch')
    require(packed(table(request.get('hu'), 42)) == packed(full['full']), 'request HU differs from sidecar HU')
    require(not request.get('hsplit') and not request.get('hbands') and not request.get('b7'), 'request has incompatible head flags')
    require(bool(request.get('hf')), 'request head features must be enabled')
    require(request.get('cutsim') == profile, 'request/profile mismatch')
    require(type(profile) is dict and set(profile) == {'module', 'sha256', 'sources', 'opts'}, 'cutsim profile fields invalid')
    module = Path(profile['module'])
    require(module.is_absolute(), 'profile module must be absolute')
    pinned({'path': str(module), 'sha256': profile['sha256']})
    opts = profile['opts']
    require(type(opts) is dict and set(opts) == {'enabled', 'law_response_fit', 'law_beta', 'law_restore_right', 'repair_head'}, 'repair profile options invalid')
    require(all(opts[k] is True for k in ('enabled', 'law_response_fit', 'law_beta', 'law_restore_right')), 'passing law flags required')
    require(opts['repair_head'] == {'enabled': values['enabled'], 'feature_spec': values['feature_spec']}, 'repair feature options mismatch')
    require(type(source['dependencies']) is list and source['dependencies'], 'dependency pins required')
    paths = []
    for pin in source['dependencies']:
        pinned(pin)
        paths.append(pin['path'])
    require(len(paths) == len(set(paths)), 'duplicate dependency paths')
    dependencies = {str(Path(pin['path']).resolve()): pin['sha256'] for pin in source['dependencies']}
    require(type(profile['sources']) is dict and profile['sources'], 'profile sources required')
    required_sources = {str(module.resolve()): profile['sha256']}
    for path, expected in profile['sources'].items():
        absolute = Path(path) if Path(path).is_absolute() else module.parent / path
        absolute = absolute.resolve()
        pinned({'path': str(absolute), 'sha256': expected})
        required_sources[str(absolute)] = expected
    required_sources[str(Path(values['feature_spec']['path']).resolve())] = SPEC_SHA
    require(all(dependencies.get(path) == expected for path, expected in required_sources.items()), 'dependency list does not cover profile/spec sources')
    binding_pin = source['parent_binding']
    full['parent_scientific_binding_verified'] = False
    if binding_pin is not None:
        binding = json.loads(pinned(binding_pin))
        require(binding.get('schema') == 'separate_repair_parent_binding_v1', 'parent binding schema')
        expected = {'native_parent_sha256': values['parent_native_sha256'],
                    'checkpoint_sha256': source['checkpoint']['sha256'],
                    'request_sha256': source['request']['sha256'],
                    'incumbent_hu_f32_sha256': full['incumbent_hu_f32_sha256']}
        require(all(binding.get(k) == v for k, v in expected.items()), 'native parent binding mismatch')
        require(binding.get('export_roundtrip_pass') is True, 'parent checkpoint/physics export has not passed roundtrip')
        pinned(binding['export_receipt'])
        full['parent_scientific_binding_verified'] = True
    full['active'] = values['enabled'] and any(v for row in full['u'] for v in row)
    return full


def literal_assignments(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            result[node.targets[0].id] = node.value
    return result


def check_parent_heads(parent, converted):
    nodes = literal_assignments(parent)
    require(not any(name.startswith('_BAND_') or name.startswith('_RPH_') for name in nodes),
            'parent already has BAND/repair heads; explicit compatible constructor required')
    for prefix, key in (('_HC_', 'hc'), ('_HCF_', 'hcf')):
        require(prefix + 'COLUMNS' in nodes and prefix + 'U' in nodes, 'missing incumbent head literals')
        names = list(ast.literal_eval(nodes[prefix + 'COLUMNS']))
        weights = ast.literal_eval(nodes[prefix + 'U'])
        expected = converted[key]
        require(names == expected['columns'], 'native incumbent column order differs from HU: ' + prefix)
        require(weights is not None and len(weights) == len(names), 'native incumbent shape differs')
        actual = table([list(row) for row in weights], len(names))
        require(packed(actual) == packed(expected['u']), 'native incumbent weights differ from HU: ' + prefix)


def signatures(source):
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'einsum':
            arg = node.args[0]
            found.add(arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                      else '<dynamic:' + ast.dump(arg, include_attributes=False) + '>')
    return sorted(found)


def construct(parent_bytes, values):
    require(type(parent_bytes) is bytes, 'parent must be exact bytes')
    require(sha(parent_bytes) == values['parent_native_sha256'], 'actual native PARENT SHA mismatch')
    converted = normalize(values)
    parent = parent_bytes.decode()
    check_parent_heads(parent, converted)
    metadata = {'schema': SCHEMA, 'parent_sha256': sha(parent_bytes), 'active': converted['active'],
                'feature_spec_sha256': SPEC_SHA, 'hu_f32_sha256': converted['hu_f32_sha256'],
                'incumbent_hu_f32_sha256': converted['incumbent_hu_f32_sha256'],
                'new_einsum_signatures': [], 'numerically_qualified': False, 'edits': [],
                'parent_scientific_binding_verified': converted['parent_scientific_binding_verified'],
                'qualification_blockers': ([] if converted['parent_scientific_binding_verified'] else
                    ['native parent scientific parameters/checkpoint/request export binding pending'])}
    if not converted['active']:
        metadata.update(output_sha256=sha(parent_bytes), identity='exact parent bytes; no wrapper/capture')
        return parent_bytes, converted, metadata
    edits = [
        ('            _E2_LAYER.clear()\n', '            _rph_begin(l)\n            _E2_LAYER.clear()\n'),
        ('        _YNT_REPAIR[0] = repair\n', '        _YNT_REPAIR[0] = repair\n        _RPH_YOUNG[0] = full_tail\n'),
        ('                diagonal = _ott_diagonal(diagonal, lam)\n',
         '                diagonal = _rph_diagonal(l, diagonal, lam, _e2_vc, _e2_s)\n'),
    ]
    rendered = parent
    for before, after in edits:
        require(rendered.count(before) == 1, 'native exact source site changed: ' + before.strip())
        rendered = rendered.replace(before, after)
        metadata['edits'].append({'before': before, 'after': after, 'count': 1})
    fragment_path = HERE / 'fragment_repair_heads_v2.py'
    fragment = fragment_path.read_text()
    require(fragment.count('_RPH_U = ()\n') == 1, 'weight literal site changed')
    fragment = fragment.replace('_RPH_U = ()\n', '_RPH_U = ' + repr(tuple(tuple(r) for r in converted['u'])) + '\n')
    require(signatures(fragment) == ['kn,kc->nc'], 'fragment signature changed')
    require('kn,kc->nc' in signatures(parent), 'projection signature absent from reviewed parent')
    rendered += '\n' + fragment
    require(signatures(rendered) == signatures(parent), 'new einsum expression introduced')
    compile(rendered, '<T48 native text build>', 'exec')
    output = rendered.encode()
    metadata.update(fragment_sha256=sha(fragment.encode()), fragment_template_sha256=sha(fragment_path.read_bytes()),
                    output_sha256=sha(output), reviewed_projection={'signature': 'kn,kc->nc',
                    'operands': [[2, 1024], [2, 128]], 'output': [1024, 128],
                    'dtype': 'float32', 'optimize': False, 'arithmetic': 'ordinary two-term dense contraction'})
    return output, converted, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent'); parser.add_argument('sidecar'); parser.add_argument('output')
    args = parser.parse_args()
    parent, sidecar, output = Path(args.parent), Path(args.sidecar), Path(args.output)
    receipt = Path(str(output) + '.build.json')
    heads = Path(str(output) + '.heads.json')
    require(output.parent.is_dir() and all(not p.exists() for p in (output, receipt, heads)), 'choose entirely new output paths')
    parent_bytes, sidecar_bytes = parent.read_bytes(), sidecar.read_bytes()
    result, converted, metadata = construct(parent_bytes, json.loads(sidecar_bytes))
    metadata.update(parent_path=str(parent.resolve()), sidecar_path=str(sidecar.resolve()), sidecar_sha256=sha(sidecar_bytes),
                    builder_sha256=sha(Path(__file__).read_bytes()))
    # Last read check binds the review result to the files actually used.
    require(parent.read_bytes() == parent_bytes and sidecar.read_bytes() == sidecar_bytes, 'inputs changed during build')
    with output.open('xb') as stream:
        stream.write(result)
    with heads.open('x') as stream:
        json.dump({k: converted[k] for k in ('incumbent_hu31', 'hc', 'hcf', 'u')}, stream, indent=2); stream.write('\n')
    with receipt.open('x') as stream:
        json.dump(metadata, stream, indent=2); stream.write('\n')
    print(json.dumps({'output': str(output), 'sha256': metadata['output_sha256'], 'active': metadata['active']}))


if __name__ == '__main__':
    main()
