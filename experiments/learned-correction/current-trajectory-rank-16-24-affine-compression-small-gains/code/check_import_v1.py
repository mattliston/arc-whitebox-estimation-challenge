"""NumPy-free stack binding proof against both independently qualified components."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
assert importlib.util.find_spec('numpy') is None
ROOT=Path('/home/arcworker/arc-workspace');HERE=Path(__file__).resolve().parent
LOCAL='/Users/matt/arc-white-box-estimation'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
resolve=lambda s:Path(str(s).replace(LOCAL,str(ROOT)))
build=read(HERE/'BUILD_v1.json')
D=HERE.parent;K=D.parent/'fable_oct19_kc_v1'
edge=D/'edge_hidden_nonlinear_v1/native_hidden8_v1'
rankdir=D/'current_rank_distill_v1/native_v1'
rank=next(x for x in read(rankdir/'CANDIDATES_v1.json')['candidates'] if x['affine_rank']==16)
stub=types.ModuleType('whestbench');stub.BaseEstimator=object;sys.modules['whestbench']=stub
def module(path,name):
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    exec(compile(path.read_text(),str(path),'exec'),m.__dict__);return m
def defs(path):return {n.name:ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def normalized(path):
    body=[n for n in ast.parse(path.read_text()).body if not (isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_FAST_RESCUE_CAP_S' and isinstance(n.value,ast.Constant) and n.value.value==1e9)]
    return ast.dump(ast.Module(body=body,type_ignores=[]))
rows=[];forms=[]
for role in ('science','release'):
    sp=resolve(build['outputs'][role]['path']);assert sha(sp)==build['outputs'][role]['sha256']
    rp=rankdir/Path(rank[role]['path']).name;assert sha(rp)==rank[role]['sha256']
    ep=edge/('candidate_msg8_edge_hidden8_nonlinear_'+role+'_v1.py')
    eb=read(edge/'BUILD_v1.json');assert sha(ep)==eb['outputs'][role]['sha256']
    s=module(sp,'stack_'+role);r=module(rp,'rank_'+role);e=module(ep,'edge_'+role)
    for name in ('_GNN_LITERAL','_VMFR_LR_LITERALS','_VMFR_CURRENT_AFFINE_RANK','_VMFR_ORIGINAL_W3_ENCODER'):
        assert getattr(s,name)==getattr(r,name),name
    for name in ('_EDGE_LITERAL','_MESSAGE_EDGE_LITERAL','_O20_EDGE_HIDDEN','_GN_EDGE','_O20_ECUT_RADICAL_A','_O20_ECUT_RADICAL_B'):
        assert getattr(s,name)==getattr(e,name),name
    assert s._VMFR_CURRENT_AFFINE_RANK==16 and s._VMFR_CM==8 and s._O20_EDGE_HIDDEN==8
    assert s._FAST_RESCUE_FRAC=={} and s._GNN_PARAMS is None and not s._E2_GNN_STATE and not s._VMFR_LR_CACHE
    assert s._EDGE_PARAMETERS is None and s._MESSAGE_EDGE_PARAMETERS is None and not s._E2_EDGE_STATE
    assert s._gn_message_edge_source_base is s._g5_message_edge_base
    assert s._O20_FLATEXTRA_ON is False
    assert s._rs_load_b_owned_ql_v1 is s._O20_FLATEXTRA_OWNED_PARENT and s._hzd_load is s._O20_FLATEXTRA_HZD_PARENT
    sd,rd,ed=defs(sp),defs(rp),defs(ep)
    for name in ('_gop2_gnn_step','_vmfr_current_lr_parameters','_vmfr_current_lr_linear'):assert sd[name]==rd[name]
    for name in ('_message_edge_source','_message_edge_parameters','_g5_message_edge_base','_ecut_raw'):assert sd[name]==ed[name]
    marker=object();called=[];old=s._gop2_gnn_step;active=s._GOP2_ACTIVE[0]
    s._gop2_gnn_step=lambda *args:(called.append(args) or marker);s._GOP2_ACTIVE[0]=True
    assert s.gnn_step(5,None,None,None,None,None) is marker and len(called)==1
    s._gop2_gnn_step=old;s._GOP2_ACTIVE[0]=active
    assert ast.unparse(ast.parse(sp.read_text()).body[-1])=='_FAST_RESCUE_FRAC = {}'
    assert 'numpy' not in sys.modules
    forms.append(normalized(sp))
    rows.append(dict(role=role,sha256=sha(sp),source_binding_pass=True,parameters_exact=True,caches_empty=True,live_dispatch=True))
assert forms[0]==forms[1]
receipt=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
    successful=True,script_sha256=sha(__file__),build_sha256=sha(HERE/'BUILD_v1.json'),cases=rows,
    numpy_free=True,no_setup=True,no_prediction=True,official_component_tiny_receipts=build['component_tiny_receipts'],
    native_stack_gain_unknown=True)
with (HERE/'NUMPY_FREE_IMPORT_v1.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))

