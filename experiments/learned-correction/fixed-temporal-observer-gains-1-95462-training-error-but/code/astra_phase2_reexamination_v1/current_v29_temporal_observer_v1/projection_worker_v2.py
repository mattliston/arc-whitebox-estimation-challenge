"""Source-only v2 adapter: bind the existing strict guard in its execution scope."""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OWNER = HERE/'projection_worker.py'
OWNER_SHA = 'd6e50c6ea61e9318864727a5ae9fd8fcb74efabea708391a8a358d6b7a078bd4'
WATCH_OWNER = HERE/'projection_watch.py'
WATCH_OWNER_SHA = '5c5b73152bd926aec8590c4bac74f15fb6f442e53669e8dd05300582e014309c'
REPAIR = HERE/'PROJECTION_V2_HARNESS_REPAIR.md'
REPAIR_SHA = '6ba99981a71ef0f2a0d59961258e419b75e989713f4a36b2ef624f3335fbd670'
ADAPTER_SHA = '3bcefbe9503e8457cc8be2d9dcf3d23effd5b0217bd44dbe9c3a3fd955d29cf0'
ADAPTER_PROTOCOL_SHA = '8730fc7b1fc11bcd8f9571667710034cca0251da9ba992caa8263d948ba7871d'
FAILED = {
    'projection_v1': (
        'ea534fb89ee81678986e5652e02464ee4cec7a4e92fc8561cf2a7b48a8bda713',
        '1bfb286b6f2d8210c76e41547b8a56568f3043cb40ae5e72c8cd542f892a7ee0',
        '33e6da7c914acf3647947d66fb42fd5dcadccd212440805f15752df308d68a53'),
    'projection_watch_v1': (
        '228e63b69246970ae77f6d1e0e70bad07a25deb4ce336858e59bbfb172376dfc',
        '359ff02069fa2a5846a589efa6cfc7c5b68b4a55204fe9f09789753b149c6551',
        '66073d5fac517f574e01ce4a0ed4884de2794d8c2f11bd29b0196c1e7861053a')}
_OWNER = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def owner():
    global _OWNER
    if _OWNER is None:
        for p,d in ((OWNER,OWNER_SHA),(WATCH_OWNER,WATCH_OWNER_SHA),(REPAIR,REPAIR_SHA)):
            require(not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest() == d,
                    'Frozen v1 owner and prospective v2 repair')
        spec = importlib.util.spec_from_file_location('projection_v2_original_worker',OWNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _OWNER = module
    return _OWNER


def environment():
    return owner().environment()


def derive_source():
    original,parent = owner().derive_source()
    edits = [
        ('projection_observe=projection_observe)',
         'projection_observe=projection_observe,require=require)'),
        ("HERE/'projection_worker.py'", "HERE/'projection_worker_v2.py'"),
        ("HERE/'projection_watch.py'", "HERE/'projection_watch_v2.py'"),
        ("HERE/'projection_v1'", "HERE/'projection_v2'"),
        ("schema='current_v29_output_projection_qualification_v1'",
         "schema='current_v29_output_projection_qualification_v2'")]
    changed = original
    for old,new in edits:
        require(changed.count(old) == 1 and new not in changed, 'Unique v2 namespace/path edit '+old)
        changed = changed.replace(old,new)
    inverse = changed
    for old,new in reversed(edits):
        require(inverse.count(new) == 1, 'Unique v2 inverse '+new)
        inverse = inverse.replace(new,old)
    require(inverse == original, 'Whole worker inverse: no paid arithmetic or guard change')
    return changed,dict(parent=parent,owner_sha256=OWNER_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        only_missing_guard_binding_and_fresh_paths_changed=True)


def execution_scope():
    base = owner()
    pins = dict(base.PROJECTION_SOURCE_PINS)
    pins.update({OWNER:OWNER_SHA,WATCH_OWNER:WATCH_OWNER_SHA,REPAIR:REPAIR_SHA})
    negative = dict(base.NEGATIVE_PINS)
    for folder,digests in FAILED.items():
        for name,digest in zip(('result.json','completion.json','finalization.json'),digests):
            negative[str(HERE/folder/name)] = digest
    return dict(__name__='projection_v2_inert',__file__=str(Path(__file__).resolve()),
                PROJECTION_SOURCE_PINS=pins,NEGATIVE_PINS=negative)


if __name__ == '__main__':
    require('--run' in sys.argv, 'Explicit --run and all six reviewed hashes required')
    source,proof = derive_source()
    scope = execution_scope(); scope['__name__'] = '__main__'
    exec(compile(source,str(Path(__file__).resolve())+':v2-namespace-repair','exec'),scope)
