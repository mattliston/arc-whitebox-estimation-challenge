"""NumPy-free binding and exact component composition proof for exploratory stack."""
import argparse,ast,hashlib,importlib.util,json,subprocess,sys,types
from pathlib import Path
assert importlib.util.find_spec('numpy') is None
ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);args=ap.parse_args()
HERE=args.folder.resolve();D=HERE.parents[2]
ROOT=Path('/home/arcworker/arc-workspace');LOCAL='/Users/matt/arc-white-box-estimation'
resolve=lambda p:Path(str(p).replace(LOCAL,str(ROOT)))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
build=read(HERE/'BUILD_v1.json')
stub=types.ModuleType('whestbench');stub.BaseEstimator=object;sys.modules['whestbench']=stub
def module(p,name):
 m=types.ModuleType(name);m.__file__=str(p);sys.modules[name]=m;exec(compile(p.read_text(),str(p),'exec'),m.__dict__);return m
def defs(p):return {n.name:ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def normalized(p):
 body=[n for n in ast.parse(p.read_text()).body if not (isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_FAST_RESCUE_CAP_S' and isinstance(n.value,ast.Constant) and n.value.value==1e9)]
 return ast.dump(ast.Module(body=body,type_ignores=[]))
rows=[];forms=[]
ib=HERE.parents[1]/'native_v1'/('local_innovation90_'+build['model'])
iready=read(ib/'LINUX_READY_v1.json')
for role in ('science','release'):
 rec=build['outputs'][role];p=resolve(rec['path']);ep=resolve(rec['edge8_parent'])
 assert sha(p)==rec['sha256'] and sha(ep)==rec['edge8_parent_sha256']
 ip=ib/Path(iready['outputs'][role]['path']).name
 assert sha(ip)==iready['outputs'][role]['sha256']
 s=module(p,'stack_'+role);e=module(ep,'edge_'+role);i=module(ip,'innovation_'+role)
 assert s._GNN_LITERAL==i._GNN_LITERAL
 for name in ('_EDGE_LITERAL','_MESSAGE_EDGE_LITERAL','_O20_EDGE_HIDDEN','_GN_EDGE','_O20_ECUT_RADICAL_A','_O20_ECUT_RADICAL_B'):
  assert getattr(s,name)==getattr(e,name),name
 assert defs(p)==defs(ep) and s._VMFR_CM==8 and s._O20_EDGE_HIDDEN==8
 assert s._FAST_RESCUE_FRAC=={} and s._GNN_PARAMS is None and not s._E2_GNN_STATE
 assert s._EDGE_PARAMETERS is None and s._MESSAGE_EDGE_PARAMETERS is None and not s._E2_EDGE_STATE
 assert s._gn_message_edge_source_base is s._g5_message_edge_base
 assert s._O20_FLATEXTRA_ON is False
 assert s._rs_load_b_owned_ql_v1 is s._O20_FLATEXTRA_OWNED_PARENT and s._hzd_load is s._O20_FLATEXTRA_HZD_PARENT
 marker=object();calls=[];old=s._gop2_gnn_step;active=s._GOP2_ACTIVE[0]
 s._gop2_gnn_step=lambda *a:(calls.append(a) or marker);s._GOP2_ACTIVE[0]=True
 assert s.gnn_step(5,None,None,None,None,None) is marker and len(calls)==1
 s._gop2_gnn_step=old;s._GOP2_ACTIVE[0]=active
 assert ast.unparse(ast.parse(p.read_text()).body[-1])=='_FAST_RESCUE_FRAC = {}'
 assert 'numpy' not in sys.modules
 forms.append(normalized(p));rows.append(dict(role=role,source_sha256=sha(p),edge_parameters_exact=True,
  innovation_head_parameters_exact=True,all_functions_equal_edge8=True,live_dispatch=True,caches_empty=True))
assert forms[0]==forms[1]
tiny=[D/'edge_hidden_nonlinear_v1/native_hidden8_v1/TINY_EDGE_v1.json',ib/'TINY_OFFICIAL_v1.json']
for path in tiny:assert read(path)['successful']
result=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),successful=True,
 numpy_free=True,cases=rows,build_sha256=sha(HERE/'BUILD_v1.json'),script_sha256=sha(__file__),
 official_component_receipts=[dict(path=str(p),sha256=sha(p)) for p in tiny],new_runtime_functions=False,
 no_gain_claim=True,no_prediction=True,no_setup=True)
with (HERE/'NUMPY_FREE_IMPORT_v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))

