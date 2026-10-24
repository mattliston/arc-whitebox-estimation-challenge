"""NumPy-free exact-parent, live binding and isolated-AST verification."""
import argparse, ast, hashlib, importlib.util, json, subprocess, sys, types
from pathlib import Path
assert importlib.util.find_spec('numpy') is None
import flopscope as fs
assert fs.__version__.startswith('0.12.3')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalized(text):
    nodes=[]
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_FAST_RESCUE_CAP_S' and isinstance(n.value,ast.Constant) and n.value.value==1e9:continue
        nodes.append(n)
    return ast.dump(ast.Module(body=nodes,type_ignores=[]))
ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
m=json.loads(a.manifest.read_text());rec=m['candidates'][0]
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('bd_padding_builder',Path(__file__).with_name('build_v1.py'));builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
rows=[];pair=[]
for role in ('science','release'):
    p=Path(rec[role]['path']);parent=Path(rec[role]['parent_path']);tail=Path(rec['append_tail'])
    assert sha(p)==rec[role]['sha256'] and sha(parent)==rec[role]['parent_sha256'] and sha(tail)==rec['append_tail_sha256']
    text=p.read_text();old=parent.read_text();assert text==old+tail.read_text()
    defs=lambda s:{n.name:n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before,after=defs(old),defs(text)
    assert set(after)-set(before)=={'_o21_birth_prepare','_o21_birth_distinct_scores','_o21_birth_refine','_o21_birth_finish'}
    assert [name for name in before if ast.dump(before[name])!=ast.dump(after[name])]==['_bd_update','_astra_fhm_core']
    bd,core,moved=builder.transform(old)
    assert bd==ast.get_source_segment(text,after['_bd_update']) and core==ast.get_source_segment(text,after['_astra_fhm_core'])
    assert moved==m['moved_born_statements']
    assert ast.unparse(ast.parse(text).body[-1])=='_FAST_RESCUE_FRAC = {}'
    mod=types.ModuleType('bd_padding_'+role);mod.__file__=str(p);sys.modules[mod.__name__]=mod;exec(compile(text,str(p),'exec'),mod.__dict__)
    assert mod._GNN_PARAMS is None and not mod._E2_MSG and not mod._E2_LAYER
    assert mod._EDGE_PARAMETERS is None and mod._MESSAGE_EDGE_PARAMETERS is None
    assert mod._FAST_RESCUE_FRAC=={} and mod._O20_FLATEXTRA_ON is False
    assert mod._rs_load_b_owned_ql_v1 is mod._O20_FLATEXTRA_OWNED_PARENT and mod._hzd_load is mod._O20_FLATEXTRA_HZD_PARENT
    assert mod._ASTRA_FHM_ACTIVE_CORE.__wrapped__ is mod._astra_fhm_core
    assert dict(mod._ASTRA_DENSE_CONTEXT)['_bd_update'] is mod._bd_update
    assert mod._bd_update.__globals__['_o21_birth_refine'] is mod._o21_birth_refine
    assert mod._astra_fhm_core.__globals__['_o21_birth_prepare'] is mod._o21_birth_prepare
    assert mod._astra_fhm_core.__globals__['_o21_birth_finish'] is mod._o21_birth_finish
    assert mod._ASTRA_FHM_ACTIVE_CORE.__closure__ is not None
    pair.append(normalized(text));rows.append(dict(role=role,sha256=sha(p),dense_context_live=True,active_core_live=True,caches_empty=True,loader_restored=True,only_declared_AST_changes=True))
assert pair[0]==pair[1]
receipt=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),successful=True,official_version=fs.__version__,numpy_free=True,no_setup=True,no_prediction=True,manifest_sha256=sha(a.manifest),script_sha256=sha(__file__),cases=rows)
with a.out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(dict(successful=True,cases=rows),indent=2))
