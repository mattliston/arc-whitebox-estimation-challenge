"""Fixed64-call common-cubic supervisor; whole-source inverse, stdlib only."""
import ast
import hashlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RD = _HERE.parent/'current_comp_receiver_directions_v1'
_RD_SHA = 'c7623198f8fb41ed1dbbe4a7ea4a0b9c7f5bd38bd0015f650d174fb4b4d87bce'


def derived_source():
    path = _RD/'watch_train.py'
    original = path.read_text()
    if hashlib.sha256(path.read_bytes()).hexdigest() != _RD_SHA:
        raise RuntimeError('Frozen RD training supervisor source changed')
    changes = (
        ('Fixed64-call receiver training supervisor', 'Fixed64-call common-cubic training supervisor'),
        ('68acc32338ef68085c68993a557ee67c64704d36acc8133126f95639e4bb592b',
         'b6435d051444b0d13443ca83d0692cb1c866b44526269d07136409555ea87503'),
        ('ad9b923d6c322585257013b293237a28b1d77826a86fb4a412a21a9bd9118b8c',
         'd90cec782ef7631acc227b1bb857f92d71284920521341764c3d182b87368d80'),
        ("d['arm']=='receiver'", "d['arm']=='common_cubic'"),
        ("d['trainable_keys']==['R']", "d['trainable_keys']==['Q'] and d['trainable_parameter_count']==65"),
        ('current_comp_receiver_directions_training_v1', 'current_comp_common_cubic_training_v1'),
        ('Complete fixed receiver-training roster', 'Complete fixed common-cubic-training roster'),
        ("':receiver-training-v1'", "':common-cubic-training-v1'"),
    )
    derived, applied = original, []
    for old, new in changes:
        count = derived.count(old)
        if not count:
            raise RuntimeError('Missing exact CC watch span: '+old)
        derived = derived.replace(old, new)
        applied.append((old, new, count))
    inverse = derived
    for old, new, count in reversed(applied):
        if inverse.count(new) != count:
            raise RuntimeError('Unexpected exact CC watch inverse: '+new)
        inverse = inverse.replace(new, old)
    if inverse != original:
        raise RuntimeError('Whole original RD training supervisor inverse failed')
    ast.parse(derived)
    return derived, dict(owner=str(path), owner_sha256=_RD_SHA,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(derived.encode()).hexdigest(),
        inverse_source_byte_exact=True, original_resource_guards_unchanged=True)


_WATCH_SOURCE, WATCH_ADAPTER_PROOF = derived_source()
exec(compile(_WATCH_SOURCE, str(_RD/'watch_train.py')+':common-cubic-watch-v1', 'exec'), globals())
