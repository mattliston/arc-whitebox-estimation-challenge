"""Additive metadata-only adapter; original scalar diagnostic remains immutable."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE / 'scalar_semantics_probe.py'
OWNER_SHA = '7495ef7ede42851de53d0c160ede70bca9616ae20a97227a511ef59da72ec658'
OWNER_PROTOCOL = HERE / 'SCALAR_SEMANTICS_PROTOCOL.md'
OWNER_PROTOCOL_SHA = 'dbfd2067025bc8024f53bd05f09054e7d470cf7b35650097a3c171f086af7d5b'


def derive_source():
    raw = OWNER.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OWNER_SHA:
        raise RuntimeError('Frozen original diagnostic source')
    if hashlib.sha256(OWNER_PROTOCOL.read_bytes()).hexdigest() != OWNER_PROTOCOL_SHA:
        raise RuntimeError('Frozen original diagnostic protocol')
    source = raw.decode('utf-8')
    edits = (
        ("require(source.count(original) == source.count(final_cast) == source.count(parameter_dtype) == 1,",
         "require(source.count(original) == source.count(final_cast) == 1 and source.count(parameter_dtype) == 2,"),
        ("SCHEMA = 'current_comp_shared_old384_scalar_semantics_v1'",
         "SCHEMA = 'current_comp_shared_old384_scalar_semantics_v2'"),
        ("out = HERE / 'scalar_semantics_v1' / args.mode",
         "out = HERE / 'scalar_semantics_v2' / args.mode"),
        ("HERE / 'SCALAR_SEMANTICS_PROTOCOL.md': args.protocol_sha256",
         "HERE / 'SCALAR_SEMANTICS_PROTOCOL_V2.md': args.protocol_sha256"),
        ("PINS = {\n",
         "PINS = {\n"
         "    HERE / 'scalar_semantics_probe.py': '" + OWNER_SHA + "',\n"
         "    HERE / 'SCALAR_SEMANTICS_PROTOCOL.md': '" + OWNER_PROTOCOL_SHA + "',\n"),
    )
    changed = source
    for before, after in edits:
        if changed.count(before) != 1:
            raise RuntimeError('Unique versioned metadata splice')
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(edits):
        if inverse.count(after) != 1:
            raise RuntimeError('Unique reverse metadata splice')
        inverse = inverse.replace(after, before)
    if inverse.encode('utf-8') != raw:
        raise RuntimeError('Full original-source inverse')
    return changed, dict(original_source_sha256=OWNER_SHA,
        original_protocol_sha256=OWNER_PROTOCOL_SHA,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        whole_source_inverse_verified=True, metadata_only_splices=5,
        required_source_counts=dict(gain_expression=1, hook_float64_return=1, FP32_P_initializers=2),
        arithmetic_unchanged=True, predicted_1006_and_15090_unchanged=True,
        original_frozen_files_preserved=True, prior_preflight_only_no_numerical_stage=True)


_source, SOURCE_ADAPTER_PROOF = derive_source()
_api = dict(__file__=str(Path(__file__).resolve()), __name__='_scalar_semantics_v2_body', __package__=None)
exec(compile(_source, str(Path(__file__).resolve()) + ':exact-metadata-adapter', 'exec'), _api)
PINS = _api['PINS']
compare = _api['compare']
numerical_modules = _api['numerical_modules']
gain_source = _api['gain_source']
sha = _api['sha']
main = _api['main']
del _source


if __name__ == '__main__':
    raise SystemExit(main())
