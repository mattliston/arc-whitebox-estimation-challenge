"""Fixed centered scalar finite-difference forward. No optimization or fitting."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['plus','minus'],required=True);ap.add_argument('--plan-sha256',required=True);ap.add_argument('--w3module',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    fd_path=HERE/'FD_PLAN_v1.json'; assert sha(fd_path)==a.plan_sha256
    fd=json.loads(fd_path.read_text()); a.rank=fd['rank']; assert a.rank==7100268
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
    req=json.loads(Path(b['request']).read_text());req.update(id=f'scalarfd_{a.arm}_{a.rank}',rank=a.rank,mode='eval',scale=1e8,keep_final=True,keep_all=True)
    claim=h/'claimed'/f"{req['id']}.w0.json";write(claim,req)
    original_run=E.run_network
    original_values={}
    def digest(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    def scalar_fd_run(*args,**kwargs):
        assert not original_values, 'One forward per process'
        original_values.update({key:digest(value) for key,value in E.T.items()})
        hu_before=digest(E.M.HEADU)
        before_dl={k:value.detach().clone() for k,value in E.DL.items()}
        before_pl={k:value.detach().clone() for k,value in E.PL.EPS.items()}
        sign=1. if a.arm=='plus' else -1.
        with w['torch'].no_grad():
            for key in ('X0','X2','s_carry','s_new'):
                E.DL[key].add_(sign*fd['plus_deltas'][key])
            for key in ('pe3','pe4'):
                E.PL.EPS[key][8:15].add_(sign*fd['plus_deltas'][key+'_L8_14'])
        actual={key:float(E.DL[key]-before_dl[key]) for key in ('X0','X2','s_carry','s_new')}
        actual.update({key+'_L8_14':(E.PL.EPS[key]-before_pl[key]).tolist() for key in ('pe3','pe4')})
        for key in E.DL:
            if key not in ('X0','X2','s_carry','s_new'):assert w['torch'].equal(E.DL[key],before_dl[key]),key
        for key in E.PL.EPS:
            if key not in ('pe3','pe4'):assert w['torch'].equal(E.PL.EPS[key],before_pl[key]),key
            else:
                assert w['torch'].equal(E.PL.EPS[key][:8],before_pl[key][:8])
                assert w['torch'].equal(E.PL.EPS[key][15:],before_pl[key][15:])
        write(h/'PERTURBATION_v1.json',dict(arm=a.arm,plan_sha256=sha(fd_path),actual_deltas=actual,temporary_process_only=True))
        answer=original_run(*args,**kwargs)
        assert {key:digest(value) for key,value in E.T.items()}==original_values
        assert digest(E.M.HEADU)==hu_before
        return answer
    E.run_network=scalar_fd_run
    w['serve'](claim)
    result=w['result_path'](h,req);error=w['error_path'](h,req)
    assert result.exists() and not error.exists(),str(error)
    r=json.loads(result.read_text());assert r['mode']=='eval'
    assert __import__('math').isfinite(r['mse']) and r['mse']>0
    assert len(cards)==1 and hasattr(cards[0],'_o18_rtt_port') and cards[0]._o18_rtt_port['pending'] is None
    write(h/'COMPLETE_v1.json',dict(rank=a.rank,arm=a.arm,mode='eval',result=str(result),result_sha256=sha(result),mse=r['mse'],seconds=r['seconds'],peak_rss=r['peak_rss'],arrays_unchanged=True,one_forward_only=True,fd_plan_sha256=sha(fd_path),harness_sha256=sha(Path(__file__)),binding_sha256=sha(D/'BINDING_v1.json')))
    print(json.dumps({'completed':a.rank,'arm':a.arm,'seconds':r['seconds'],'peak_rss':r['peak_rss']}))
if __name__=='__main__':main()
