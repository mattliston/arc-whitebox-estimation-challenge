"""Inert v2 path adapter; the original complete resource loop is unchanged."""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE/'projection_watch.py'
BASE_SHA = '5c5b73152bd926aec8590c4bac74f15fb6f442e53669e8dd05300582e014309c'
WORKER_V1 = HERE/'projection_worker.py'
WORKER_V1_SHA = 'd6e50c6ea61e9318864727a5ae9fd8fcb74efabea708391a8a358d6b7a078bd4'
REPAIR = HERE/'PROJECTION_V2_HARNESS_REPAIR.md'
REPAIR_SHA = '6ba99981a71ef0f2a0d59961258e419b75e989713f4a36b2ef624f3335fbd670'
_BASE = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def base():
    global _BASE
    if _BASE is None:
        for path,digest in ((BASE,BASE_SHA),(WORKER_V1,WORKER_V1_SHA),(REPAIR,REPAIR_SHA)):
            require(not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                    'Frozen preceding worker/watch and repair scope')
        spec = importlib.util.spec_from_file_location('projection_watch_v2_owner',BASE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BASE = module
    return _BASE


def derive_source():
    original,parent = base().derive_source()
    edits = [
        ("HERE/'projection_watch.py'", "HERE/'projection_watch_v2.py'",1),
        ("HERE/'projection_worker.py'", "HERE/'projection_worker_v2.py'",3),
        ("HERE/'projection_v1'", "HERE/'projection_v2'",1),
        ("HERE/'projection_watch_v1'", "HERE/'projection_watch_v2'",1),
        ("current_v29_output_projection_qualification_v1", "current_v29_output_projection_qualification_v2",1),
        ("current_v29_output_projection_qualification_watch_v1", "current_v29_output_projection_qualification_watch_v2",1),
        ("V29_WATCH_OWNER:V29_WATCH_SHA,V29_WATCH_SPEC:V29_WATCH_SPEC_SHA}",
         "V29_WATCH_OWNER:V29_WATCH_SHA,V29_WATCH_SPEC:V29_WATCH_SPEC_SHA,\n"
         "        REPAIR:REPAIR_SHA,BASE:BASE_SHA,WORKER_V1:WORKER_V1_SHA}",1)]
    changed = original
    for old,new,count in edits:
        require(changed.count(old) == count and new not in changed,'Unique v2 watch path/owner edit '+old)
        changed = changed.replace(old,new)
    inverse = changed
    for old,new,count in reversed(edits):
        require(inverse.count(new) == count,'Unique v2 watch inverse '+new)
        inverse = inverse.replace(new,old)
    require(inverse == original,'Whole source inverse, complete resource loop unchanged')
    return changed,dict(parent=parent,owner_sha256=BASE_SHA,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        numerical_and_resource_gates_unchanged=True)


def execution_scope():
    value = base()
    names = ('OWNER','OWNER_SHA','V29_WATCH_OWNER','V29_WATCH_SHA','V29_WATCH_SPEC','V29_WATCH_SPEC_SHA')
    scope = {name:getattr(value,name) for name in names}
    scope.update(__name__='projection_watch_v2_inert',__file__=str(Path(__file__).resolve()),
        REPAIR=REPAIR,REPAIR_SHA=REPAIR_SHA,BASE=BASE,BASE_SHA=BASE_SHA,
        WORKER_V1=WORKER_V1,WORKER_V1_SHA=WORKER_V1_SHA)
    return scope


if __name__ == '__main__':
    require('--run' in sys.argv,'Explicit --run --numerical-slot-clear and all six hashes required')
    source,proof = derive_source()
    scope = execution_scope(); scope['__name__'] = '__main__'
    exec(compile(source,str(Path(__file__).resolve())+':v2-fresh-paths','exec'),scope)
