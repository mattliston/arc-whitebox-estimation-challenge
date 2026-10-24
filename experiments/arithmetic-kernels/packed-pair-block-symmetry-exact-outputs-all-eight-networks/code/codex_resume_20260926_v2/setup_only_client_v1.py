"""Numpy-free, fresh-client setup receipt for the production RPC boundary."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import types
from flopscope import _connection

assert importlib.util.find_spec('numpy') is None
candidate=Path(sys.argv[1]).resolve();target=Path(sys.argv[2])
assert not target.exists()
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
mod=types.ModuleType('setup_candidate');mod.__file__=str(candidate)
sys.modules[mod.__name__]=mod
exec(compile(candidate.read_text(),str(candidate),'exec'),mod.__dict__)
counter=[0]
original=_connection.Connection.send_recv
def counted(self,request):
    counter[0]+=1
    return original(self,request)
_connection.Connection.send_recv=counted
est=mod.Estimator()
ctx=types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0)
start=time.monotonic();est.setup(ctx);elapsed=time.monotonic()-start
record=dict(candidate=str(candidate),sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            setup_s=elapsed,requests=counter[0],pb_log=[list(x) for x in mod._PB_LOG])
target.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
