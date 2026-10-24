"""Prepare immutable requests only after root's exact PREREG_v2.json exists.

Never launches a process or removes HELD_BY_ROOT. `smoke` creates one identity
request; `screen` requires that identity result; `confirm` requires both complete
fresh16 arms. Existing requests and decisions are compared, never overwritten.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from table_weightfit_v3 import prereg, smoke, screen, HERE, O


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen(path,data):
    if path.exists():
        assert json.loads(path.read_text()) == data, str(path)
    else:
        with path.open('x') as f:
            json.dump(data,f,indent=2,allow_nan=False)


def main(phase):
    config=prereg()
    manifest=json.loads((HERE/'harness_manifest_v3.json').read_text())
    for name, expected in manifest['files'].items():
        assert digest(HERE/name)==expected, 'Source changed: '+name
    base_path=sorted((O/'physicist/t141_k4l_ext/claimed').glob('ph141_r7100584_c.w*.json'))[0]
    base=json.loads(base_path.read_text())
    assert base['rank']==7100584
    assert digest(base_path)==config['baseline_request_sha256']
    sources={p.name:digest(p) for p in (HERE/'cutsim').glob('*.py')}
    target=HERE/'cutsim/torch_cutsim_s46_v11.py'
    if phase=='smoke':
        seeds, arms=[7100584],['z']
    elif phase=='screen':
        assert smoke()['bitexact'], 'C no-op must pass before fresh arms'
        seeds, arms=config['screen_seeds'],['z','q','h']
    else:
        decision=screen()
        assert decision['complete'] and decision['winner']
        frozen(HERE/'screen_decision_negative_v1.json',decision)
        seeds, arms=config['confirmation_seeds'],[decision['winner']]
    run=HERE/('run_'+phase)
    assert (run/'HELD_BY_ROOT').exists(), 'Preparation is held; root controls launch separately'
    # Exact truth provenance is frozen before a request is made. Only tiny scalar
    # fields are loaded, never weights, predictions, or truth target arrays.
    import numpy as np
    truth=[]
    for seed in seeds:
        path=O/'analyst/fresh'/f'truth_seed{seed}.npz'
        with np.load(path,allow_pickle=False) as z:
            N=int(z['N']);actual_seed=int(z['seed']);weight_sha=str(z['weights_sha256'])
        assert actual_seed==seed and N>=15990784
        truth.append(dict(seed=seed,N=N,weights_sha256=weight_sha,truth_sha256=digest(path)))
    frozen(run/'truth_manifest_v1.json',truth)
    allowed=','.join(map(str,seeds))
    allow_path=run/'allowed_seeds.txt'
    if allow_path.exists():
        assert allow_path.read_text()==allowed
    else:
        with allow_path.open('x') as f:f.write(allowed)
    hashes={}
    for arm in arms:
        for seed in seeds:
            request=copy.deepcopy(base)
            rid=f'wfit_{arm}_r{seed}'
            request.update(id=rid,rank=seed)
            request['cutsim'].update(module=str(target),sha256=sources[target.name],sources=sources)
            request['cutsim']['opts']['law_fit_blend']=config['arms'][arm]
            queued=run/'queue'/(rid+'.json')
            existing=[queued] if queued.exists() else list((run/'claimed').glob(rid+'.w*.json'))
            if existing:
                assert len(existing)==1 and json.loads(existing[0].read_text())==request
                hashes[rid]=digest(existing[0])
            else:
                assert not (run/'results'/(rid+'.json')).exists()
                tmp=run/'queue'/('.tmp_'+uuid.uuid4().hex)
                with tmp.open('x') as f:json.dump(request,f)
                try:os.link(tmp,queued)
                finally:tmp.unlink()
                hashes[rid]=digest(queued)
    frozen(run/'requests_manifest_v1.json',dict(prereg_sha256=digest(HERE/'PREREG_v2.json'),requests=hashes))
    print(json.dumps(dict(phase=phase,requests=len(hashes),held=True,evaluations_launched=0)))


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('smoke','screen','confirm')
    main(sys.argv[1])
