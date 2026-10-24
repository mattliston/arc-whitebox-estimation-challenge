"""Inert filename/schema successor of the full bounded neural supervisor."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
OWNER=HERE/'watch_neural_qualification.py'
OWNER_SHA='53863b5236010ea0ac72d3c5cd78b78dc798f2bfe204485316b1533bf42b1878'
_OWNER=None


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def owner():
    global _OWNER
    if _OWNER is None:
        require(not OWNER.is_symlink() and hashlib.sha256(OWNER.read_bytes()).hexdigest()==OWNER_SHA,
            'Frozen complete neural-v1 supervisor')
        spec=importlib.util.spec_from_file_location('neural_watch_v1_metadata_owner',OWNER)
        _OWNER=importlib.util.module_from_spec(spec); spec.loader.exec_module(_OWNER)
    return _OWNER


def owner_pins():
    return {OWNER:OWNER_SHA}


def derive_source():
    parent=owner(); original,parent_proof=parent.derive_source()
    source=parent.rewrite(original,[
        ("SCHEMA = 'current_v29_temporal_observer_neural_qualification_watch_v1'",
         "SCHEMA = 'current_v29_temporal_observer_neural_qualification_watch_v2'",1),
        ("HERE/'watch_neural_qualification.py'","HERE/'watch_neural_qualification_v2.py'",1),
        ("HERE/'neural_qualification.py'","HERE/'neural_qualification_v2.py'",3),
        ("HERE/'NEURAL_QUALIFICATION_PROTOCOL.md'","HERE/'NEURAL_QUALIFICATION_V2_PROTOCOL.md'",1),
        ("HERE/'NEURAL_QUALIFICATION_WATCH_SPEC.md'","HERE/'NEURAL_QUALIFICATION_V2_WATCH_SPEC.md'",1),
        ("_neural.OWNER:_neural.OWNER_SHA}","_neural.OWNER:_neural.OWNER_SHA,**_v2.owner_pins()}",1),
        ("HERE/'neural_qualification_v1'","HERE/'neural_qualification_v2'",1),
        ("HERE/'neural_qualification_watch_v1'","HERE/'neural_qualification_watch_v2'",1),
    ])
    return source,dict(owner_sha256=OWNER_SHA,parent=parent_proof,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),filename_schema_owner_changes_only=True,
        unchanged_180s_12GB_16GB_zero_swap_120s_progress_1s_observation_and_publication=True,
        all_eleven_evaluations_fresh=True)


def execution_scope():
    return dict(__name__='neural_qualification_watch_v2_inert',__file__=str(Path(__file__).resolve()),
        _neural=types.SimpleNamespace(**dict(vars(owner()))),_v2=types.SimpleNamespace(**dict(globals())))


if __name__=='__main__':
    require('--run' in sys.argv, 'Inert; explicit --run --mps-slot-clear and actual hashes required')
    source,proof=derive_source(); scope=execution_scope()
    scope.update(__name__='__main__',_neural_watch_proof=proof)
    exec(compile(source,str(OWNER)+':neural-v2-path-inverse','exec'),scope)
