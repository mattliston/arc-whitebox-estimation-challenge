"""DRAFT: current-source evidence adapter over the unchanged release/credential core."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_comp_shared_old384_v1/release.py'
OWNER_SHA='054c1e7704a12e7bb9145b8d302fae053bbd540e80b91fbc45e1c439bb5bda62'
CANDIDATE_SHA='c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
SOURCE_FLAGS=('evidence_source','native40_source','native40_watch_source','native40_protocol',
    'native40_watch_spec','score40_source','score40_watch_source','score40_protocol','score40_watch_spec')
PINS={OWNER:OWNER_SHA,
    OWNER.parent/'RELEASE_DESIGN_20260912.md':'454501f232570983e84c4c2651413482e2141c8c2a2880e2db0d37f8cb3c9326'}


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def evidence(args,bindings,scope):
    path=HERE/'release_evidence.py'
    scope['bind'](bindings,path,args.evidence_source_sha256)
    spec=importlib.util.spec_from_file_location('gain_release_actual_evidence',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.evidence(args,bindings,scope)


def derive_source():
    raw=OWNER.read_bytes()
    require(not OWNER.is_symlink() and hashlib.sha256(raw).hexdigest()==OWNER_SHA,'Frozen original release core')
    original=raw.decode();tree=ast.parse(original);lines=original.splitlines(keepends=True)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evidence')
    old_evidence=''.join(lines[node.lineno-1:node.end_lineno])
    stage=next(n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='STAGES' for t in n.targets))
    old_stages=''.join(lines[stage.lineno-1:stage.end_lineno])
    new_stages="""STAGES = {
    'native40':('native40_v2',3000,8_000_000_000),
    'native40_watch':('native40_watch_v2',3000,8_000_000_000),
    'score40':('score40_v1',120,1_000_000_000),
    'score40_watch':('score40_watch_v1',120,1_000_000_000),
    'native_panel':('native_panel_v1',1200,8_000_000_000),
    'native_panel_watch':('native_panel_watch_v1',1200,8_000_000_000),
    'score_panel':('score_panel_v1',120,2_000_000_000),
    'score_panel_watch':('score_panel_watch_v1',120,2_000_000_000),
    'production_watch':('production_watch_v1',600,8_000_000_000),
}
"""
    edits=[
        (old_evidence,'def evidence(args, bindings):\n    return _current.evidence(args,bindings,globals())\n',1),
        (old_stages,new_stages,1),
        ("CANDIDATE_SHA = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'",
         "CANDIDATE_SHA = '"+CANDIDATE_SHA+"'",1),
        ("shared_old384_4d98fef2_whest016.tar.gz","shared_gain_slice_c2c2ffad_whest016.tar.gz",1),
        ("DESCRIPTION = ('Phase 2 shared old-cumulant basis: fixed current heads, '",
         "DESCRIPTION = ('Phase 2 shared old-cumulant basis with dtype-stable gain: fixed current heads, '",1),
        ("PINS = {\n","PINS = {\n    **_current.PINS,\n",1),
        ("for key in STAGES:\n","for key in (*STAGES,*_current.SOURCE_FLAGS):\n",1),
        ("        env.pop('PYTHONOPTIMIZE',None)\n",
         "        env.pop('PYTHONOPTIMIZE',None)\n        env.pop('PYTHONPATH',None)\n"
         "        env.pop('FLOPSCOPE_SERVER_URL',None)\n        env['RAILS_HOST']='www.aicrowd.com'\n",1),
        ('RELEASE_DESIGN_20260912.md','RELEASE_DESIGN.md',2),
        ("schema='current_comp_shared_old384_release_v1'","schema='current_comp_shared_gain_slice_release_v1'",1),
        ("local_native40_adjusted=records['score40']['summary']['shared']['adjusted'])",
         "local_native40_adjusted=records['score40']['summary']['shared']['adjusted'],\n"
         "            release_fidelity_proof=records['release_fidelity_proof'],\n"
         "            current_release_source_proof=_current_release_proof)",1),
    ]
    source=original;history=[]
    for before,after,count in edits:
        require(source.count(before)==count,'Exact current release span '+before[:90])
        cursor=0;positions=[]
        for _ in range(count):
            at=source.index(before,cursor);positions.append(at);cursor=at+len(before)
        for at in reversed(positions):
            source=source[:at]+after+source[at+len(before):];history.append((at,before,after))
    inverse=source
    for at,before,after in reversed(history):
        require(inverse[at:at+len(after)]==after,'Exact current release inverse span')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    require(inverse==original,'Whole original resource/CLI/credential/archive/retry/publication core restored')
    compile(source,'gain-current-release-derived','exec')
    return source,dict(owner_sha256=OWNER_SHA,whole_source_inverse_verified=True,
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),current_source_and_evidence_adaptation_only=True,
        original_resource_CLI_archive_credential_immutable_attempt_and_no_retry_core=True)


def execution_scope():
    return dict(__name__='gain_current_release_inert',__file__=str(Path(__file__).resolve()),
        _current=types.SimpleNamespace(**dict(globals())))


if __name__=='__main__':
    require('--package' in sys.argv or '--submit' in sys.argv,'Explicit reviewed package or submit phase only')
    source,proof=derive_source();scope=execution_scope()
    scope.update(__name__='__main__',_current_release_proof=proof)
    exec(compile(source,str(OWNER)+':current-evidence-release','exec'),scope)
