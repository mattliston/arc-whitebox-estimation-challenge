"""NumPy-free complete source binding check for direct edge8+complement+squared-hidden readout."""
import argparse,ast,base64,hashlib,json,subprocess,sys,types,zlib,importlib.util
from pathlib import Path
assert importlib.util.find_spec('numpy') is None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser()
for k in ('build','out'):ap.add_argument('--'+k,type=Path,required=True)
a=ap.parse_args();build=json.loads(a.build.read_text())
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
def literal(src):
 nodes=[n for n in ast.parse(src).body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_GNN_LITERAL']
 return ast.literal_eval(nodes[-1].value)
def definitions(src):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
cases=[];suffixes=[]
for role,rec in build['outputs'].items():
 path=Path(rec['path']);parent=Path(rec['parent_path']);assert sha(path)==rec['sha256'] and sha(parent)==rec['parent_sha256']
 src=path.read_text();old=parent.read_text();assert src.startswith(old);suffixes.append(src[len(old):])
 p,q=literal(old),literal(src);assert set(q)==set(p)|{'square_head'}
 for key in p:
  assert p[key][0]==q[key][0]
  if key not in ('head','head_b','hidden_head'):assert zlib.decompress(base64.b64decode(p[key][1]))==zlib.decompress(base64.b64decode(q[key][1]))
 assert q['head'][0]==(16,64) and q['head_b'][0]==(16,) and q['hidden_head'][0]==(16,128) and q['square_head'][0]==(16,128)
 for key,nbyte in [('head',256),('head_b',4)]:
  assert zlib.decompress(base64.b64decode(p[key][1]))[:nbyte]==zlib.decompress(base64.b64decode(q[key][1]))[:nbyte]
 assert zlib.decompress(base64.b64decode(q['hidden_head'][1]))[:512]==bytes(512)
 assert zlib.decompress(base64.b64decode(q['square_head'][1]))[:512]==bytes(512)
 od,nd=definitions(old),definitions(src);assert set(nd)==set(od)|{'_sq90_square_head'}
 assert {k:v for k,v in nd.items() if k not in ('_gop2_gnn_step','_sq90_square_head')}=={k:v for k,v in od.items() if k!='_gop2_gnn_step'}
 assert ast.unparse(ast.parse(src).body[-1])=='_FAST_RESCUE_FRAC = {}'
 mod=types.ModuleType('hidden_complement_'+role);mod.__file__=str(path);sys.modules[mod.__name__]=mod
 exec(compile(src,str(path),'exec'),mod.__dict__)
 assert mod._GNN_PARAMS is None and not mod._E2_GNN_STATE and mod._FAST_RESCUE_FRAC=={}
 assert mod._EDGE_PARAMETERS is None and mod._MESSAGE_EDGE_PARAMETERS is None
 assert mod._O20_EDGE_HIDDEN==8 and not mod._E2_EDGE_STATE
 assert mod._VMFR_CM==8 and mod._gn_message_edge_source_base is mod._g5_message_edge_base
 assert mod._O20_FLATEXTRA_ON is False and mod._hzd_load is mod._O20_FLATEXTRA_HZD_PARENT
 assert mod._rs_load_b_owned_ql_v1 is mod._O20_FLATEXTRA_OWNED_PARENT
 assert '_gop2_gnn_step' in mod.gnn_step.__code__.co_names
 assert mod.gnn_step.__globals__['_gop2_gnn_step'] is mod._gop2_gnn_step
 assert '_hc90_hidden_head' in mod._gop2_gnn_step.__code__.co_names
 assert mod._gop2_gnn_step.__globals__['_hc90_hidden_head'] is mod._hc90_hidden_head
 assert '_sq90_square_head' in mod._gop2_gnn_step.__code__.co_names
 assert mod._gop2_gnn_step.__globals__['_sq90_square_head'] is mod._sq90_square_head
 original=types.ModuleType('parent_squared_'+role);original.__file__=str(parent);sys.modules[original.__name__]=original
 exec(compile(old,str(parent),'exec'),original.__dict__)
 for name in ('_EDGE_LITERAL','_MESSAGE_EDGE_LITERAL','_O20_EDGE_HIDDEN','_GN_EDGE','_O20_ECUT_RADICAL_A','_O20_ECUT_RADICAL_B'):
  assert getattr(mod,name)==getattr(original,name),name
 marker=object();calls=[];previous=mod._gop2_gnn_step;active=mod._GOP2_ACTIVE[0]
 mod._gop2_gnn_step=lambda *args:(calls.append(args) or marker);mod._GOP2_ACTIVE[0]=True
 assert mod.gnn_step(5,None,None,None,None,None) is marker and len(calls)==1
 mod._gop2_gnn_step=previous;mod._GOP2_ACTIVE[0]=active
 assert mod._GNN_LITERAL==q
 cases.append(dict(role=role,source_sha256=sha(path),head_and_bias_plus_hidden_and_squared_readouts_only=True,edge8_literals_unchanged=True,cold_caches_empty=True,loader_restored=True,live_global_bindings=True))
assert suffixes[0]==suffixes[1]
r=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),successful=True,numpy_free=True,cases=cases,no_setup=True,no_prediction=True,script_sha256=sha(__file__))
with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(successful=True,numpy_free=True,live_global_bindings=True)))

