"""Configure private children after original configure_cutsim(spec)."""
from pathlib import Path
import importlib.util
D = Path(__file__).resolve().parent
def load(name):
    spec = importlib.util.spec_from_file_location('_o18_current_'+name, D/(name+'.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
def configure(runtime_module):
    push=load('torch_push_current95c_rtt_v2')
    row=load('torch_rowcuts_rtt_private_v5')
    port=load('torch_rtt_current95c_v2')
    row._O18_RTT_MODULE=port;row._O18_PUSH_MODULE=push
    runtime_module._CHILDREN['torch_push_native_v2.py']=push
    runtime_module._CHILDREN['torch_rowcuts_v15.py']=row
    return push,row,port
