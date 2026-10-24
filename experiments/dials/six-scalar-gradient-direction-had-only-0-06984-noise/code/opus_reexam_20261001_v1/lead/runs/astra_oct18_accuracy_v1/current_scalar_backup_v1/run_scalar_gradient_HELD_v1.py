"""HELD current-physics six-coordinate scalar gradient; carrier derivative is approximate. No update."""
import argparse, hashlib, json, os, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PORT=HERE.parent/'current_torch_rtt_port_v1'
D=PORT.parent/'post336935_training_audit_v1'
O=D.parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,choices=[7100268],required=True);ap.add_argument('--w3module',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    b=json.loads((D/'BINDING_v1.json').read_text())
    for k in ('source','params','request','worker'):assert sha(b[k])==b[k+'_sha256']
    h=a.output.resolve();h.mkdir()
    for n in ('queue','claimed','results'): (h/n).mkdir()
    for n in ('TRAINER_GENERATED_ALLOW_FILE','TRAINER_TRUTH_PLAN','TRAINER_TRUTH_PLAN_SHA256'):os.environ.pop(n,None)
    allow=h/'ALLOW.txt';allow.write_text(str(a.rank)+'\n')
    os.environ['TRAINER_GENERATED_ALLOW_FILE']=str(allow)
    os.environ['TRAINER_GENERATED_MIN_N']='7995392'
    worker=Path(b['worker']);sys.path.insert(0,str(worker.parent));sys.argv=[str(worker),str(h),'0']
    source=worker.read_text();cut="say(event='start', run=str(RUN)";assert source.count(cut)==1
    w={'__name__':'_fixed_gap_worker','__file__':str(worker)}
    exec(compile(source[:source.index(cut)],str(worker),'exec'),w)
    E=w['E'];assert E.THREADS==w['torch'].get_num_threads()==w['torch'].get_num_interop_threads()==1
    import importlib.util
    def load(name,path):
        sp=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);return mod
    adapter=load('_private_rtt_configure',PORT/'configure_private_v8.py')
    W3=load('_private_current_w3',a.w3module)
    W3.install_worker_module(E.M,w['torch'])
    current_m_install=E.M.install
    def current_install(cd):
        runtime=E._otr_cut_runtime_v1
        module=runtime['module']
        if module is None:return current_m_install(cd)
        module.install(cd,opts=runtime['spec'].get('opts'))
        return module.install_m(cd,current_m_install)
    E.M.install=current_install
    original_configure=w['configure_cutsim']
    def configure(spec):
        original_configure(spec)
        children=adapter.configure(E._otr_cut_runtime_v1['module'])
        assert children[0].RANKS=={7:384,10:384,13:384}
    w['configure_cutsim']=configure
    old_card=E.LC.card
    cards=[]
    def scalar_card(*args,**kwargs):
        cd=old_card(*args,**kwargs);cd.knobs['bfix_gamma']=.119;cards.append(cd);return cd
    E.LC.card=scalar_card
    req=json.loads(Path(b['request']).read_text());req.update(id=f'currentgradient_{a.rank}',rank=a.rank,mode='grad',scale=1e8,keep_final=False,keep_all=False)
    claim=h/'claimed'/f"{req['id']}.w0.json";write(claim,req)
    # Same numerical model, with four global scalar leaves and two PL families.
    # Only the uniform L8..14 sums of pe3/pe4 are candidate coordinates. The
    # remaining vector entries are diagnostic outputs, never selectable arms.
    import inspect
    wanted_dials=('X0','X2','s_carry','s_new')
    wanted_pl=('pe3','pe4')
    run_source=inspect.getsource(E.run_network)
    changes={
        "record['hu_grads'] = M.HEADU.grad.clone()": "record['hu_grads'] = torch.zeros_like(M.HEADU)",
        "dial_grads = {key: DL[key].grad.clone() for key in DIAL_LR}": "dial_grads = {key: DL[key].grad.clone() for key in ('X0','X2','s_carry','s_new')}",
        "record['pl_grads'] = PL.collect()": "record['pl_grads'] = {key: value for key,value in PL.collect().items() if key in ('pe3','pe4')}"}
    for old,new in changes.items():
        assert run_source.count(old)==1,old
        run_source=run_source.replace(old,new)
    namespace=dict(E.run_network.__globals__)
    exec(compile(run_source,'<private scalar-gradient reporting>','exec'),namespace)
    run_selected=namespace['run_network']
    original_values={}
    def tensor_items(value):
        if isinstance(value,w['torch'].Tensor):yield value
        elif isinstance(value,dict):
            for item in value.values():yield from tensor_items(item)
        elif isinstance(value,(tuple,list)):
            for item in value:yield from tensor_items(item)
    def digest(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    def selective_run(*args,**kwargs):
        # Freeze after serve has loaded every request leaf, including HEADU.
        for collection in (E.T,E.DL,E.PL.EPS,E.M.HEADU,getattr(E.M,'EDGEU',None),getattr(E.M,'EDGEXS',None),getattr(E.M,'RG_W',None)):
            for value in tensor_items(collection):value.requires_grad_(False);value.grad=None
        for key in wanted_dials:E.DL[key].requires_grad_(True)
        for key in wanted_pl:E.PL.EPS[key].requires_grad_(True)
        E.TRAINABLE=[]
        original_values.update({key:digest(value) for key,value in E.T.items()})
        namespace.update(vars(E))
        return run_selected(*args,**kwargs)
    E.run_network=selective_run
    write(h/'DERIVATIVE_SCOPE_v1.json',dict(requested_arrays=[],requested_dials=wanted_dials,
        requested_pl=wanted_pl,uniform_pl_band=list(range(8,15)),frozen_hu=True,
        parameter_values_changed=False,partial_derivatives='Approximate frozen-carrier/basis derivative; mandatory directional finite difference before any batch or point.',no_optimizer=True))
    w['serve'](claim)
    result=w['result_path'](h,req);error=w['error_path'](h,req)
    assert result.exists() and not error.exists(),str(error)
    r=w['torch'].load(result,map_location='cpu',weights_only=False);assert r['mode']=='grad'
    assert r['grads']=={} and set(r['dial_grads'])==set(wanted_dials) and set(r['pl_grads'])==set(wanted_pl)
    assert {key:digest(value) for key,value in E.T.items()}==original_values
    assert all(value.grad is None and not value.requires_grad for value in E.T.values())
    coords={key:float(r['dial_grads'][key]) for key in wanted_dials}
    for key in wanted_pl:
        assert tuple(r['pl_grads'][key].shape)==(16,)
        coords[key+'_L8_14']=float(r['pl_grads'][key][8:15].sum())
    assert all(__import__('math').isfinite(x) for x in coords.values())
    write(h/'GRADIENT_SUMMARY_v1.json',dict(mse=r['mse'],mse_layers=r['mse_layers'],seconds=r['seconds'],peak_rss=r['peak_rss'],coordinate_gradients_scaled=coords,scale=r['scale'],parameters_updated=False,derivative_approximate=True,requires_directional_finite_difference=True))
    write(h/'COMPLETE_v1.json',dict(rank=a.rank,binding_sha256=sha(D/'BINDING_v1.json'),result=str(result),result_sha256=sha(result),request_sha256=sha(claim),current_scalars=b['current_scalars'],one_backward_only=True,parameters_updated=False,no_optimizer=True,arrays_unchanged=True))
    assert len(cards)==1 and hasattr(cards[0],'_o18_rtt_port')
    assert cards[0]._o18_rtt_port['pending'] is None
    write(h/'PORT_STATS_v1.json',dict(reads=cards[0]._o18_rtt_port['reads'],w3module=str(a.w3module),w3sha=sha(a.w3module),private_sources={p.name:sha(p) for p in PORT.glob('*.py')},one_backward_only=True))
    print(json.dumps({'completed':a.rank,'seconds':r['seconds'],'peak_rss':r['peak_rss']}))
if __name__=='__main__':main()
