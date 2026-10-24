"""One frozen-current-physics training-seed gradient; no optimizer or parameter update."""
import argparse, hashlib, json, os, sys
from pathlib import Path
PORT=Path(__file__).resolve().parent
D=PORT.parent/'post336935_training_audit_v1'
O=D.parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,choices=[7100268],required=True);ap.add_argument('--w3module',type=Path,required=True);a=ap.parse_args()
    b=json.loads((D/'BINDING_v1.json').read_text())
    for k in ('source','params','request','worker'):assert sha(b[k])==b[k+'_sha256']
    h=PORT/f'gradient_{a.rank}_v3';h.mkdir()
    for n in ('queue','claimed','results'): (h/n).mkdir()
    for n in ('TRAINER_GENERATED_ALLOW_FILE','TRAINER_TRUTH_PLAN','TRAINER_TRUTH_PLAN_SHA256'):os.environ.pop(n,None)
    os.environ['TRAINER_GENERATED_ALLOW_FILE']=str(PORT/'GRADIENT_ALLOW_v1.txt')
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
    adapter=load('_private_rtt_configure',PORT/'configure_private_v5.py')
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
    w['serve'](claim)
    result=w['result_path'](h,req);error=w['error_path'](h,req)
    assert result.exists() and not error.exists(),str(error)
    r=w['torch'].load(result,map_location='cpu',weights_only=False);assert r['mode']=='grad'
    gradient_groups={name:r[name] for name in ('grads','dial_grads','pl_grads')}
    gradient_summary={}
    for name,group in gradient_groups.items():
        assert group
        gradient_summary[name]={}
        for key,value in group.items():
            value=w['torch'].as_tensor(value)
            assert bool(w['torch'].isfinite(value).all()),(name,key)
            gradient_summary[name][key]={'shape':list(value.shape),'norm':float(w['torch'].linalg.vector_norm(value.to(w['torch'].float64)))}
    assert any(x['norm']>0 for x in gradient_summary['grads'].values())
    write(h/'GRADIENT_SUMMARY_v1.json',dict(mse=r['mse'],mse_layers=r['mse_layers'],seconds=r['seconds'],peak_rss=r['peak_rss'],gradient_summary=gradient_summary,parameters_updated=False))
    write(h/'COMPLETE_v1.json',dict(rank=a.rank,binding_sha256=sha(D/'BINDING_v1.json'),result=str(result),result_sha256=sha(result),request_sha256=sha(claim),current_scalars=b['current_scalars'],one_backward_only=True,parameters_updated=False,no_optimizer=True))
    assert len(cards)==1 and hasattr(cards[0],'_o18_rtt_port')
    assert cards[0]._o18_rtt_port['pending'] is None
    write(h/'PORT_STATS_v1.json',dict(reads=cards[0]._o18_rtt_port['reads'],w3module=str(a.w3module),w3sha=sha(a.w3module),private_sources={p.name:sha(p) for p in PORT.glob('*.py')},one_backward_only=True))
    print(json.dumps({'completed':a.rank,'seconds':r['seconds'],'peak_rss':r['peak_rss']}))
if __name__=='__main__':main()
