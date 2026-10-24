"""Inert raw-host observer adapter: preserve qualified projected prefix outputs."""
import hashlib
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
OWNER = HERE/'model.py'
OWNER_SHA = '59564d394f37758bb85805099b2f32ff9a6576825339b561c689e688f9c087ef'
OLD = '    served = torch.cat((host[:15], terminal[None, :]), dim=0)'
NEW = '    served = torch.cat((torch.clamp(host[:15], min=0.), terminal[None, :]), dim=0)'


def derive_source():
    raw = OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest() != OWNER_SHA:
        raise RuntimeError('Frozen original observer source')
    original = raw.decode()
    if original.count(OLD) != 1 or NEW in original:
        raise RuntimeError('Unique unprojected prefix output')
    changed = original.replace(OLD,NEW)
    if changed.replace(NEW,OLD).encode() != raw:
        raise RuntimeError('Whole model inverse')
    return changed,dict(owner_sha256=OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        raw_host_semantics_preserved=True,only_prefix_output_projected=True,
        parameter_and_training_rule_changes=False)


def load_model():
    source,proof = derive_source()
    module = types.ModuleType('v29_projected_observer_model')
    module.__file__ = str(OWNER)
    exec(compile(source,str(OWNER)+':projected-prefix','exec'),vars(module))
    return module,proof
