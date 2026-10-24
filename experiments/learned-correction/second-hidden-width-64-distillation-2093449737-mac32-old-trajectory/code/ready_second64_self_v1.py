"""Qualify the frozen current-message8 self-distilled second-hidden candidate.
Only import/source/official-tiny validation; no native benchmark is launched.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

def main():
    v=Path(__file__).resolve().parent
    out=v/'fit_second64_self_native_v1'
    manifest=out/'CANDIDATES_v1.json'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
    def write(path,obj):
        with path.open('x') as f:
            json.dump(obj,f,indent=2,allow_nan=False)
            f.write('\n')
    assert importlib.util.find_spec('numpy') is None
    assert sha(manifest)=='5a1005dfaa1c7536fb8a115cd91db6610364eed1f21716cb90fe2d38cf67f39d'
    m=json.loads(manifest.read_text())
    tiny=json.loads((out/'TINY_OFFICIAL_v5.json').read_text())
    assert tiny['status']=='TINY_OFFICIAL_FUNCTIONS_PASS_NO_NETWORK_SCORE'
    assert tiny['manifest_sha256']==sha(manifest)
    assert all(x['repeat_byte_exact'] for x in tiny['cases'])
    c,=m['candidates']
    assert c['name']=='msg8_second64_selfteacher'
    pins={'science':'fb50200f408e65fa8f6e54fde120b543692cab7c297325121f6f99a7a19d3fa3','release':'a63bb173d4dec271b2a9152fa9cc0608712210cee1adedcbac23c95151e0b63c'}
    stub=types.ModuleType('whestbench')
    stub.BaseEstimator=type('BaseEstimator',(),{})
    sys.modules['whestbench']=stub
    rows=[]
    for role,pin in pins.items():
        p=Path(c[role]['path'])
        assert sha(p)==c[role]['sha256']==pin
        mod=types.ModuleType('second64_self_'+role)
        mod.__file__=str(p)
        sys.modules[mod.__name__]=mod
        exec(compile(p.read_text(),str(p),'exec'),mod.__dict__)
        assert mod._FAST_RESCUE_FRAC=={} and mod._GNN_PARAMS is None and not mod._E2_GNN_STATE
        assert mod._O20_FLATEXTRA_ON is False
        assert mod._rs_load_b_owned_ql_v1 is mod._O20_FLATEXTRA_OWNED_PARENT
        assert mod._hzd_load is mod._O20_FLATEXTRA_HZD_PARENT
        rows.append(dict(role=role,source_sha256=pin,full_client_import='PASS',coldcaches=True,no_governor=True,loader_bindings_restored=True))
    write(out/'FULL_IMPORT_v1.json',dict(utc=utc(),imports=rows,numpy_free=True))
    old=v/'PLAN_SECOND64_V7_v1.json'
    assert sha(old)=='f68da500ba7184646231b57a763b65b34f3528ae6179323b21998f04c7de107f'
    plan=json.loads(old.read_text())
    p=Path(c['science']['path'])
    plan['candidates'][c['name']]=dict(path=str(p.relative_to('/home/arcworker/arc-workspace')),bytes=p.stat().st_size,role='frozen_candidate',sha256=sha(p),effective_sha256=sha(p))
    plan['utc']=utc()
    pp=v/'PLAN_SECOND64_SELF_MARKLINE_v1.json'
    write(pp,plan)
    ready=dict(utc=utc(),status='TINY_FULLIMPORT_READY_NO_NATIVE_LAUNCH',plan=str(pp),plan_sha256=sha(pp),candidates=[dict(name=c['name'],science=c['science'],release=c['release'])],validation_limitation='Teacher-validation3 inputs are off-policy; native same-host paired scoring required.')
    write(v/'SECOND64_SELF_READY_v1.json',ready)
    print(json.dumps(ready,indent=2))
if __name__=='__main__':
    main()

