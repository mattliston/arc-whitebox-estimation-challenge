"""Additive RemoteScalar snapshot API repair; all measured arithmetic is unchanged."""
import ast
import hashlib
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
OWNER = HERE / 'scalar_semantics_probe_v2.py'
OWNER_SHA = 'e2645d36e17cc3db5beff1e39a4da2c911df405f65127a594827e0c6f193d32a'
OWNER_PROTOCOL = HERE / 'SCALAR_SEMANTICS_PROTOCOL_V2.md'
OWNER_PROTOCOL_SHA = 'e583fae3c64a4b11bcabee21ccab6225320db47a262ed0baf929ea6b6ab03316'
REMOTE_SOURCE = HERE.parent.parent / 'data_gate/.client-venv/lib/python3.11/site-packages/flopscope/_remote_array.py'
REMOTE_SHA = '8c74bfdcd08502d1b53daac4aa0b8c24809aaa1408c17e6178deb83a9876fd54'


def derive_source():
    for path, digest in ((OWNER, OWNER_SHA), (OWNER_PROTOCOL, OWNER_PROTOCOL_SHA), (REMOTE_SOURCE, REMOTE_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen v2 owner and actual RemoteScalar source')
    classes = [n for n in ast.parse(REMOTE_SOURCE.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'RemoteScalar']
    if len(classes) != 1:
        raise RuntimeError('Unique actual scalar class, not RemoteArray')
    methods = {n.name: n for n in classes[0].body if isinstance(n, ast.FunctionDef)}
    expected = ast.parse('def __float__(self) -> float:\n    return float(self._value)\n').body[0]
    if 'item' in methods or '__float__' not in methods or ast.dump(methods['__float__']) != ast.dump(expected):
        raise RuntimeError('Exact public scalar float conversion; no item method')
    old = runpy.run_path(str(OWNER), run_name='_scalar_v3_frozen_v2_owner')
    source, prior = old['derive_source']()
    edits = (
        ("raw = struct.pack('<f' if dtype == 'float32' else '<d', value.item())",
         "raw = struct.pack('<f' if dtype == 'float32' else '<d', float(value))"),
        ("SCHEMA = 'current_comp_shared_old384_scalar_semantics_v2'",
         "SCHEMA = 'current_comp_shared_old384_scalar_semantics_v3'"),
        ("out = HERE / 'scalar_semantics_v2' / args.mode",
         "out = HERE / 'scalar_semantics_v3' / args.mode"),
        ("HERE / 'SCALAR_SEMANTICS_PROTOCOL_V2.md': args.protocol_sha256",
         "HERE / 'SCALAR_SEMANTICS_PROTOCOL_V3.md': args.protocol_sha256"),
        ("PINS = {\n", "PINS = {\n"
         "    HERE / 'scalar_semantics_probe_v2.py': '" + OWNER_SHA + "',\n"
         "    HERE / 'SCALAR_SEMANTICS_PROTOCOL_V2.md': '" + OWNER_PROTOCOL_SHA + "',\n"),
    )
    changed = source
    for before, after in edits:
        if changed.count(before) != 1:
            raise RuntimeError('Unique scalar v3 snapshot/metadata edit')
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(edits):
        if inverse.count(after) != 1:
            raise RuntimeError('Unique scalar v3 inverse')
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('Full inverse to frozen v2 body')
    return changed, dict(prior=prior, owner_sha256=OWNER_SHA, owner_protocol_sha256=OWNER_PROTOCOL_SHA,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(), whole_source_inverse_verified=True,
        actual_RemoteScalar_float_API_verified=True, snapshot_only_API_repair=True,
        measured_arithmetic_and_predicted_tariffs_unchanged=True, old_v2_negative_receipts_preserved=True)


_source, SOURCE_ADAPTER_PROOF = derive_source()
_api = dict(__file__=str(Path(__file__).resolve()), __name__='_scalar_semantics_v3_body', __package__=None)
exec(compile(_source, str(Path(__file__).resolve()) + ':exact-snapshot-adapter', 'exec'), _api)
PINS = _api['PINS']
compare = _api['compare']
numerical_modules = _api['numerical_modules']
gain_source = _api['gain_source']
sha = _api['sha']
main = _api['main']
del _source


if __name__ == '__main__':
    raise SystemExit(main())
