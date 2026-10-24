"""NumPy-free full source, exact constant change and live bindings."""
import argparse,ast,base64,hashlib,json,subprocess,sys,types,zlib,importlib.util
from pathlib import Path
assert importlib.util.find_spec('numpy') is None
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser()
for key in ('build','out'):ap.add_argument('--'+key,type=Path,required=True)
a=ap.parse_args();b=json.loads(a.build.read_text())
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
def literal(src):
    nodes=[n for n in ast.parse(src).body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_GNN_LITERAL']
    return ast.literal_eval(nodes[-1].value)
cases=[];suffixes=[]
for role,rec in b['outputs'].items():
    path=Path(rec['path']);parent=Path(rec['parent_path']);assert sha(path)==rec['sha256'] and sha(parent)==rec['parent_sha256']
    src=path.read_text();old=parent.read_text();assert src.startswith(old);suffixes.append(src[len(old):])
    p,q=literal(old),literal(src);assert p.keys()==q.keys()
    for key in p:
        assert p[key][0]==q[key][0]
        if key!='head':assert zlib.decompress(base64.b64decode(p[key][1]))==zlib.decompress(base64.b64decode(q[key][1]))
    assert p['head'][0]==(16,64)
    assert zlib.decompress(base64.b64decode(p['head'][1]))[:256]==zlib.decompress(base64.b64decode(q['head'][1]))[:256]
    assert ast.unparse(ast.parse(src).body[-1])=='_FAST_RESCUE_FRAC = {}'
    mod=types.ModuleType('innovation_'+role);mod.__file__=str(path);sys.modules[mod.__name__]=mod
    exec(compile(src,str(path),'exec'),mod.__dict__)
    assert mod._GNN_PARAMS is None and not mod._E2_GNN_STATE and mod._FAST_RESCUE_FRAC=={}
    assert mod._EDGE_PARAMETERS is None and mod._MESSAGE_EDGE_PARAMETERS is None
    assert mod._VMFR_CM==8 and mod._gn_message_edge_source_base is mod._g5_message_edge_base
    assert mod._O20_FLATEXTRA_ON is False and mod._hzd_load is mod._O20_FLATEXTRA_HZD_PARENT
    assert mod._rs_load_b_owned_ql_v1 is mod._O20_FLATEXTRA_OWNED_PARENT
    cases.append(dict(role=role,sha256=sha(path),head_only=True,cold_caches_empty=True,loader_restored=True))
assert suffixes[0]==suffixes[1]
r=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
    successful=True,numpy_free=True,cases=cases,no_setup=True,no_prediction=True,script_sha256=sha(__file__))
with a.out.open('x') as out:json.dump(r,out,indent=2);out.write('\n')
print(json.dumps(dict(successful=True,numpy_free=True)))

