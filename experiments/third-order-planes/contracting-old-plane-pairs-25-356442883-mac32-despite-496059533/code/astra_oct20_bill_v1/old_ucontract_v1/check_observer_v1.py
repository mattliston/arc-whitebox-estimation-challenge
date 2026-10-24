"""Small mocked profile lifecycle; no estimator import or native prediction."""
import hashlib,importlib.util,json,struct,subprocess,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;PARENT=HERE.parents[1]/'astra_oct20_vm_frontier_v1/msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py'
sp=importlib.util.spec_from_file_location('su_hook',HERE/'old_su_hook_v1.py');hook=importlib.util.module_from_spec(sp);sp.loader.exec_module(hook)
class Fake:
    def __init__(self,shape,integer=False):self.shape=shape;self.integer=integer
    def _fetch_data(self):
        size=1
        for n in self.shape:size*=n
        return (b'\x00'*(size*(8 if self.integer else 4)),self.shape,'int64'if self.integer else'float32')
env=dict(Fake=Fake);exec(compile('\n'*14042+'def _ubx_product(left,right,uv,leg):\n    return Fake((left.shape[0],right.shape[1]))\n','<mock corresponding-call-site>','exec'),env)
mod=types.SimpleNamespace(__file__=str(PARENT),_FAST_RESCUE_FRAC={},_ubx_product=env['_ubx_product'],_UVP_CUR=[10]);before=sys.getprofile()
out=HERE/'observer_fixture_v1';out.mkdir(exist_ok=False);observer=hook.install(mod,out,{})
def context(layer,part):
    live_indices=Fake((7,),True);base=(Fake((256,8)),Fake((384,8)),[(0,8)],256);join=(Fake((256,4)),Fake((384,4)),[(0,4)],256)
    state=dict(pre_basis=Fake((12,384)),columns=16,rank=384,lazy=None,uvp=('m',base))
    uv=base
    if layer>=13:
        state['lazy']=dict(map=Fake((384,384)));state['uvp']=('l',base,join)
        if part=='join':uv=join
    mod._UVP_CUR[0]=layer;P=uv[0].shape[1]
    result=mod._ubx_product(Fake((7,P)),Fake((P,256)),uv,'ut')
    assert result.shape==(7,256)
for rank in(35,150):
    observer.begin(rank,'scored')
    for layer in range(10,15):
        for part in(('materialized',)if layer<13 else('base','join')):context(layer,part)
receipt=observer.finish();assert sys.getprofile()is before and not observer.pending and len(receipt['files'])==14
for entry in receipt['files']:
    r=json.loads((out/'old_su_capture_v1'/entry['path']).read_text());assert 'Su'in r and 'FuT'in r and 'Dv'not in r and r['native_inner']['shape']==[7,256]
result=dict(utc=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),successful=True,hook_sha256=hashlib.sha256((HERE/'old_su_hook_v1.py').read_bytes()).hexdigest(),records=14,profile_restored=True,common_output_width=256,no_estimator_import=True,no_prediction=True,no_truth=True)
(HERE/'OBSERVER_CHECK_v1.json').open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
