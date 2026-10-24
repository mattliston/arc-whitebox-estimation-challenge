"""HELD request preparer. Root registration required; never launches workers."""
import copy,json,os,sys,uuid
from pathlib import Path
from table_rank_redistribution_v3 import HERE,O,WEIGHTED,HARNESS,ARMS,sha,prereg,identity,effects,screen
def frozen(p,x):
    if p.exists():assert json.loads(p.read_text())==x,str(p)
    else:
        with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def main(phase):
    d=prereg()
    if phase=='smoke':seeds,arms=[7100584],{'zero':'zero'}
    elif phase=='effects':
        assert identity()['passed'];seeds,arms=[7100584],ARMS
    elif phase=='screen':
        assert effects()['passed'];seeds,arms=d['screen_seeds'],ARMS
        frozen(HERE/'CONTROL_IDENTITY_v1.json',dict(identity=identity(),effects=effects()))
    else:
        decision=screen();assert decision['complete'] and decision['survivors']
        frozen(HERE/'screen_decision_v1.json',decision)
        seeds,arms=d['confirmation_seeds'],{a:ARMS[a] for a in decision['survivors']}
    run=HERE/('run_'+phase)
    for sub in ('queue','claimed','results'):(run/sub).mkdir(parents=True,exist_ok=True)
    # A phase cannot be prepared or amended after a root launcher activated it.
    assert not list(run.glob('HELD_BY_ROOT.launched_*'))
    for marker in ('STOP','HELD_BY_ROOT'):
        p=run/marker
        if not p.exists():
            with p.open('x') as f:f.write('Held; root controls launch separately.\n')
    basep=O/'physicist/t141_k4l_ext/claimed/ph141_r7100584_c.w0.json'
    assert sha(basep)=='fb4866e86e55138adb33d1f097a9d5a6c8a8597413c47340eb1ae2bd72a8b22a'
    base=json.loads(basep.read_text());truth=[]
    assert base['cutsim']['opts']['law_restore_right'] is True
    assert base['params']==d['baseline_params']
    import numpy as np
    for seed in seeds:
        p=O/'analyst/fresh'/f'truth_seed{seed}.npz'
        with np.load(p,allow_pickle=False) as z:
            N=int(z['N']);actual=int(z['seed']);weights_sha=str(z['weights_sha256'])
        assert actual==seed and N>=15990784
        truth.append(dict(seed=seed,N=N,weights_sha256=weights_sha,truth_sha256=sha(p)))
    frozen(run/'truth_manifest_v1.json',truth)
    allow=','.join(map(str,seeds));p=run/'allowed_seeds.txt'
    if p.exists():assert p.read_text()==allow
    else:
        with p.open('x') as f:f.write(allow)
    hashes={}
    for arm,suffix in arms.items():
        target=HARNESS/('cutsim_'+suffix)/'torch_cutsim_s46_v11.py'
        sources={p.name:sha(p) for p in target.parent.glob('*.py')}
        for seed in seeds:
            req=copy.deepcopy(base);rid=f'rdist_{arm}_r{seed}';req.update(id=rid,rank=seed)
            if phase in ('smoke','effects'):req['keep_all']=True
            req['cutsim'].update(module=str(target),sha256=sources[target.name],sources=sources)
            req['cutsim']['opts'].update(ru=256,law_fit_blend=0.0)
            q=run/'queue'/(rid+'.json');existing=([q] if q.exists() else list((run/'claimed').glob(rid+'.w*.json')))
            if existing:
                assert len(existing)==1 and json.loads(existing[0].read_text())==req;hashes[rid]=sha(existing[0])
            else:
                assert not (run/'results'/(rid+'.json')).exists()
                tmp=run/'queue'/('.tmp_'+uuid.uuid4().hex)
                with tmp.open('x') as f:json.dump(req,f)
                try:os.link(tmp,q)
                finally:tmp.unlink()
                hashes[rid]=sha(q)
    frozen(run/'requests_manifest_v1.json',dict(prereg_sha256=sha(HERE/'PREREG_v1.json'),requests=hashes))
    print(json.dumps(dict(phase=phase,requests=len(hashes),held=True,evaluations_launched=0)))
if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('smoke','effects','screen','confirm')
    main(sys.argv[1])
