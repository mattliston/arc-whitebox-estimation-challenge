"""DRAFT: exact bounded neural supervisor adapted to the one624-rollout proposal."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
OWNER=HERE/'watch_neural_qualification.py'
OWNER_SHA='53863b5236010ea0ac72d3c5cd78b78dc798f2bfe204485316b1533bf42b1878'
EXTRA_FLAGS=('neural-source','neural-watch-source','neural-protocol','neural-watch-spec',
    'neural','neural-watch','check-capture','check-capture-watch','joint-inventory',
    'capture-worker','capture-watch-source','capture-protocol','capture-watch-spec',
    'capture','capture-watch','projection-qualification','projection-qualification-watch')
_OWNER=None


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def owner():
    global _OWNER
    if _OWNER is None:
        require(not OWNER.is_symlink() and hashlib.sha256(OWNER.read_bytes()).hexdigest()==OWNER_SHA,'Exact original full neural supervisor')
        spec=importlib.util.spec_from_file_location('training_watch_neural_owner',OWNER)
        _OWNER=importlib.util.module_from_spec(spec); spec.loader.exec_module(_OWNER)
    return _OWNER


def check_roster(path,stage):
    require(stage=='train' and not path.is_symlink() and path.stat().st_size<1<<20,'Fixed bounded TRAIN roster')
    value=json.loads(path.read_text())
    train=(35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58); check=(37,39,42,48)
    require(value['schema']=='current_v29_temporal_observer_training_roster_v1'
        and value['seed']==93091230 and value['generator']=='PCG64DXSM' and len(value['orders'])==30
        and all(len(order)==20 and all(type(r) is int for r in order) and set(order)==set(train) for order in value['orders']),
        'Fixed generator/seed and30 complete disjoint-from-CHECK TRAIN permutations')
    expected=[]
    for epoch,order in enumerate(value['orders']):
        for i,rank in enumerate(order):
            expected.append(dict(graph_index=len(expected),rank=rank,kind='train_backward',device='mps',epoch=epoch,update=epoch*5+i//4))
    for kind,ranks in (('terminal_train_forward',train),('terminal_check_forward',check)):
        for rank in ranks:
            expected.append(dict(graph_index=len(expected),rank=rank,kind=kind,device='mps',epoch=None,update=None))
    require(value['entries']==expected and len(expected)==624,'Exact600 immediate backwards then20+4 terminal cached forwards')
    return expected


def derive_source():
    parent=owner(); original,prior=parent.derive_source()
    source=parent.rewrite(original,[
        ("SCHEMA = 'current_v29_temporal_observer_neural_qualification_watch_v1'",
         "SCHEMA = 'current_v29_temporal_observer_training_watch_v1'",1),
        ("parser.set_defaults(stage='qualify')","parser.set_defaults(stage='train')",1),
        ('    limit,expected = 180.,11','    limit,expected = 600.,624',1),
        ('_neural.CAPTURE_FLAGS','_training.EXTRA_FLAGS',3),
        ('return _neural.check_roster(path,stage)','return _training.check_roster(path,stage)',1),
        ("HERE/'watch_neural_qualification.py'","HERE/'watch_train.py'",1),
        ("HERE/'neural_qualification.py'","HERE/'train.py'",3),
        ("HERE/'NEURAL_QUALIFICATION_PROTOCOL.md'","HERE/'TRAIN_PROTOCOL.md'",1),
        ("HERE/'NEURAL_QUALIFICATION_WATCH_SPEC.md'","HERE/'TRAIN_WATCH_SPEC.md'",1),
        ("_neural.OWNER:_neural.OWNER_SHA}","_neural.OWNER:_neural.OWNER_SHA,_training.OWNER:_training.OWNER_SHA}",1),
        ("HERE/'neural_qualification_v1'","HERE/'training_v1'",1),
        ("HERE/'neural_qualification_watch_v1'","HERE/'training_watch_v1'",1),
        ("(path.is_relative_to(P2) or path in _neural.EXTERNAL_PINS)",
         "(path.is_relative_to(P2) or path in _neural.EXTERNAL_PINS\n"
         "        or path == _neural.SITE/'torch/nn/utils/clip_grad.py')",1),
        ('        api.verify_result(record,worker_args,admitted,tick)\n        union = {}',
         '        api.verify_result(record,worker_args,admitted,tick)\n'
         "        for path,digest in record['target_bindings'].items():\n"
         "            api.hash_target(path,digest,admitted['targets'],tick)\n"
         "        result.update(TRAIN_target_bytes_hashed=True,TRAIN_target_values_decoded=False,\n"
         "            validation_targets_accessed=False,target_bindings=dict(record['target_bindings']))\n"
         '        union = {}',1),
        ('foreign_numerical_workers_checked=True,neural_execution_qualified=True,',
         'foreign_numerical_workers_checked=True,proposal_execution_complete=True,',1),
    ])
    return source,dict(owner_sha256=OWNER_SHA,parent=prior,whole_source_inverse=True,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),fixed_neural_rollouts=624,
        backwards=600,AdamW_updates=150,whole_cap_s=600,
        unchanged_12GB_16GB_zero_swap_120s_progress_1s_observations_and_sealed_tail=True,
        exact20_TRAIN_target_byte_hasher_separate_from_nontarget_hasher=True,physical_model_calls=0)


def execution_scope():
    return dict(__name__='training_watch_inert',__file__=str(Path(__file__).resolve()),
        _neural=types.SimpleNamespace(**dict(vars(owner()))),_training=types.SimpleNamespace(**dict(globals())))


if __name__=='__main__':
    require('--run' in sys.argv,'Inert; explicit --run --mps-slot-clear and actual hashes required')
    source,proof=derive_source(); scope=execution_scope()
    scope.update(__name__='__main__',_neural_watch_proof=proof)
    exec(compile(source,str(OWNER)+':one-fixed-training-source-inverse','exec'),scope)
