"""Passive evidence-only inverse adapter to the unexecuted v2 source."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE/'qualification_v2.py'
PARENT_SHA = '978fd3d034a06705312f2be4cd6a0b6317e57997b4537ddfada3e5297bac9682'
PINS = {PARENT: PARENT_SHA,
        HERE/'PROTOCOL_V2.md': '4a7ec42195f9713bf7c5c15c34f58f459e14e714947d0a29b57f058687ae86b2'}
_PARENT = None


def parent():
    global _PARENT
    for path, digest in PINS.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen v2 source/protocol changed: '+str(path))
    if _PARENT is None:
        spec = importlib.util.spec_from_file_location('receiver_qualification_v2_owner', PARENT)
        _PARENT = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_PARENT)
    return _PARENT


def derived_source():
    owner = parent()
    owner.require(not (HERE/'qualification_v2').exists(), 'V2 remains source-only, not a selected numerical run')
    original, prior_proof = owner.derived_source()
    source = original
    anchor = "                            served_jvp = save(f'jvp_{device}_{rank:04d}_depth{card.depth}', vector)\n"
    changes = (
        ("OUT = HERE/'qualification_v2'", "OUT = HERE/'qualification_v3'"),
        ("schema='current_comp_receiver_birth_qualification_v2'", "schema='current_comp_receiver_birth_qualification_v3'"),
        ("HERE/'PROTOCOL_V2.md': args.protocol_sha256", "HERE/'PROTOCOL_V3.md': args.protocol_sha256"),
        ("HERE/'watch_qualification_v2.py': args.watcher_sha256", "HERE/'watch_qualification_v3.py': args.watcher_sha256"),
        (anchor, anchor+
         "                            served_jvp['served_primal'] = save(\n"
         "                                f'jvp_{device}_{rank:04d}_depth{card.depth}_served_primal', array(yp))\n"),
    )
    for old, new in changes:
        owner.require(source.count(old) == 1, 'One passive v3 source span')
        source = source.replace(old, new)
    inverse = source
    for old, new in reversed(changes):
        owner.require(inverse.count(new) == 1, 'One passive v3 inverse span')
        inverse = inverse.replace(new, old)
    owner.require(inverse == original, 'Exact inverse to frozen v2 derived source')
    return source, dict(parent_adapter_source_sha256=PARENT_SHA, parent_proof=prior_proof,
        parent_derived_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        inverse_source_byte_exact=True, actual_served_primal_and_tangent_saved=True,
        model_and_scalar_and_reverse_unchanged=True, numerical_gates_and_roster_unchanged=True,
        original_qualification_still_failed=True, parent_v2_numerically_unexecuted=True)


def main():
    owner = parent()
    source, proof = derived_source()
    env = dict(__file__=str(Path(__file__).resolve()), __name__='receiver_qualification_v3_derived',
               ADAPTER_PROOF=proof, ADAPTER_SOURCE_PINS={**owner.SOURCE_PINS, **PINS},
               ADAPTER_INPUT_PINS=owner.INPUT_PINS,
               ADAPTER_MODEL_SHA=owner.SOURCE_PINS[HERE/'model.py'])
    exec(compile(source, str(__file__)+':passive_evidence_adapter', 'exec'), env)
    return env['main']()


if __name__ == '__main__':
    raise SystemExit(main())
