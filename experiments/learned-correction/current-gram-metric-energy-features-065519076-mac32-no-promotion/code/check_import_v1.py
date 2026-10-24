"""NumPy-free complete imports and live energy-only binding verification."""
import argparse
import array
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import zlib

assert importlib.util.find_spec('numpy') is None
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
p = argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
build=json.loads(a.build.read_text());stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
cases=[];tails=[]
for role,rec in build['outputs'].items():
    path,parent=Path(rec['path']),Path(rec['parent_path'])
    assert sha(path)==rec['sha256'] and sha(parent)==rec['parent_sha256']
    source,original=path.read_text(),parent.read_text();assert source.startswith(original);tails.append(source[len(original):])
    mod=types.ModuleType('energy23_metric_'+role);mod.__file__=str(path);sys.modules[mod.__name__]=mod;exec(compile(source,str(path),'exec'),mod.__dict__)
    shape,blob=mod._GNN_LITERAL['enc0_w'];assert tuple(shape)==(128,78)
    data=array.array('f');data.frombytes(zlib.decompress(base64.b64decode(blob,validate=True)))
    expected=[[[float(i==j)for j in range(8)]for i in range(128)]]
    expected.extend([[list(data[i*78+start:i*78+start+8])for i in range(128)]for start in (62,70)])
    assert mod._ECM8_ENCODER_LITERAL==expected and mod._ECM8_ROOT is None
    assert '_ecm8_energy_contract' in mod._gop2_gnn_step.__code__.co_names
    assert '_gop2_contract' not in mod._gop2_gnn_step.__code__.co_names
    assert '_gop2_gnn_step' in mod.gnn_step.__code__.co_names
    assert '_gop2_contract' in mod._gop2_message_product.__code__.co_names
    assert '_w3r_product' in mod._gop2_message_product.__code__.co_names
    assert mod._gop2_contract.__code__.co_firstlineno==17702 and mod._w3r_metric.__code__.co_firstlineno==25463
    assert mod._GNN_PARAMS is None and not mod._E2_GNN_STATE and not mod._E2_MSG
    assert mod._FAST_RESCUE_FRAC=={} and mod._VMFR_CM==8 and mod._GOP2_ON
    assert mod._gn_message_edge_source_base is mod._g5_message_edge_base
    assert mod._O20_FLATEXTRA_ON is False and mod._hzd_load is mod._O20_FLATEXTRA_HZD_PARENT
    assert mod._rs_load_b_owned_ql_v1 is mod._O20_FLATEXTRA_OWNED_PARENT
    cases.append(dict(role=role,source_sha256=sha(path),E1_identity=True,E23_exact_current_encoder=True,
        energy_only_live_binding=True,M2_W3_untouched=True,caches_cold=True,loader_bindings_preserved=True))
assert tails[0]==tails[1]
result=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),successful=True,numpy_free=True,no_setup=True,no_prediction=True,cases=cases,script_sha256=sha(__file__))
with a.out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(successful=True,numpy_free=True)))
