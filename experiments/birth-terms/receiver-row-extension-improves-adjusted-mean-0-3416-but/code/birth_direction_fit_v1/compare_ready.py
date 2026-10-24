"""Enforce numerical, same-head, native, gradient and score-fidelity gates."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if not args.check: parser.print_help();return
    from qualify_model import sha,write
    assert not (HERE/'ready_v1.json').exists()
    import numpy as np
    files=[HERE/'model.py',HERE/'PROTOCOL.md',HERE/'qualify_model.py',HERE/'qualify_wide_v2.py',
           HERE/'fnp/build_v2.py',HERE/'fnp/check_small_v2.py',HERE/'NATIVE_SCOPE_CLARIFICATION.md',
           Path(__file__),HERE/'small_v1/result.json',HERE/'cpu_v1/result.json',HERE/'mps_v1/result.json',
           HERE/'fnp_small_v2/result.json',HERE/'native_zero_v1/result.json',HERE/'native_nonzero_v1/result.json']
    pins={str(p):sha(p) for p in files}
    certificates={key:json.loads((HERE/path).read_text()) for key,path in
        [('small','small_v1/result.json'),('cpu','cpu_v1/result.json'),('mps','mps_v1/result.json'),
         ('fnp_small','fnp_small_v2/result.json'),('native_zero','native_zero_v1/result.json'),
         ('native_nonzero','native_nonzero_v1/result.json')]}
    for name,cert in certificates.items():
        assert cert['passed'],name
        if name.startswith('native') or name=='fnp_small':
            folder={'native_zero':'native_zero_v1','native_nonzero':'native_nonzero_v1',
                    'fnp_small':'fnp_small_v2'}[name]
            completion=HERE/folder/'completion.json'
            end=json.loads(completion.read_text())
            assert end['passed'] and end['result_sha256']==sha(HERE/folder/'result.json')
            pins[str(completion)]=sha(completion)
        if 'sources' in cert:
            for p,h in cert['sources'].items():
                assert sha(p)==h,p
                pins[p]=h
        for p,value in cert.get('source_bindings',{}).items():
            if isinstance(value,dict): p,h=value['path'],value['sha256']
            else: h=value
            assert sha(p)==h,p
            pins[p]=h
        if name.startswith('native'):
            for p,h in cert['artifacts'].items(): assert sha(p)==h,p
            assert cert['trials'][0]['output_bytes_sha256']==cert['trials'][-1]['output_bytes_sha256']
            assert cert['initialization_flops']==1383913786
            build_path=Path(cert['source']).with_name('build.json')
            build=json.loads(build_path.read_text())
            theta=np.asarray(json.loads((HERE/'fixture_nonzero_theta.json').read_text())['theta'],dtype=np.float32)
            expected=np.array([[math.tanh(float(x)) for x in row] for row in theta],dtype=np.float32)
            if name=='native_zero': expected=np.zeros_like(expected)
            assert np.asarray(build['coefficient_literal'],dtype=np.float32).tobytes()==expected.tobytes(),name
    checks=[];sealed=[]
    oldroot=PARENT/'unused_materializations_v1/native'
    assert sha(oldroot/'result.json')=='a9ccecaa3c0ca79f070af3c0961a6a952f08f4d9712e82a3d6af4d9a0d1d0951'
    pins[str(oldroot/'result.json')]=sha(oldroot/'result.json')
    old=json.loads((oldroot/'result.json').read_text())
    for rank,call in ((35,0),(100,1)):
        grads=[]
        for dev in ('cpu','mps'):
            path=HERE/f'{dev}_v1/{dev}_{rank}_gradient.npy'
            pins[str(path)]=sha(path);grads.append(np.load(path).astype(float))
        grad_error=float(np.linalg.norm(grads[1]-grads[0])/np.linalg.norm(grads[0]))
        assert grad_error<=.05,(rank,grad_error)
        item=dict(rank=rank,gradient_relative_error=grad_error,actions={})
        for action in ('zero','nonzero'):
            native=certificates['native_'+action]['trials'][call]
            paths={dev:HERE/f'{dev}_v1/{dev}_{rank}_{action}.npy' for dev in ('cpu','mps')}
            paths['native']=Path(native['output'])
            for dev in ('cpu','mps'):
                recorded=next(x for x in certificates[dev]['checks']
                              if x.get('rank')==rank and x.get('action')==action)
                assert sha(paths[dev])==recorded['prediction_sha256']
            arrays={dev:np.load(p) for dev,p in paths.items()}
            for p in paths.values(): pins[str(p)]=sha(p)
            differences={dev:float(np.max(np.abs(arrays[dev].astype(float)-arrays['native'].astype(float))))
                         for dev in ('cpu','mps')}
            assert max(differences.values())<=1e-5,(rank,action,differences)
            if action=='zero':
                oldname=f'call_{call:02d}_rank{rank}_output.npy'
                assert sha(oldroot/oldname)==old['artifacts'][oldname]['sha256']
                pins[str(oldroot/oldname)]=sha(oldroot/oldname)
                assert arrays['native'].tobytes()==np.load(oldroot/oldname).tobytes()
            else:
                zero_path=Path(certificates['native_zero']['trials'][call]['output'])
                assert arrays['native'][0].tobytes()==np.load(zero_path)[0].tobytes()
            item['actions'][action]=dict(max_absolute_differences=differences,
                complete_native_flops=native['flops'])
            sealed.append((rank,action,arrays))
        checks.append(item)
    # All model outputs/bills have been authenticated BEFORE mean-label access.
    write(HERE/'readiness_predictions_sealed_v1.json',dict(sources=pins,
        utc=datetime.now(timezone.utc).isoformat(),truth_accessed=False))
    truth_hashes={35:'ec47c040198070cfe936126d18d24d02a94a6fef1da14795173c23d199e05f61',
                  100:'637a63d46ac5480be50cca3f0021a1574eef80fa580ce98f1a8cc69b0b86450c'}
    for rank,action,arrays in sealed:
        truthpath=PARENT.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        assert sha(truthpath)==truth_hashes[rank]
        pins[str(truthpath)]=truth_hashes[rank]
        with np.load(truthpath) as src: truth=src['mc_post'][-1].astype(float)
        errors={dev:float(np.mean((array[-1].astype(float)-truth)**2)) for dev,array in arrays.items()}
        ratios={dev:errors[dev]/errors['native'] for dev in ('cpu','mps')}
        assert all(abs(ratio-1)<=.01 for ratio in ratios.values()),(rank,action,ratios)
        row=next(x for x in checks if x['rank']==rank)['actions'][action]
        row.update(raw_errors=errors,raw_fidelity_ratios=ratios,truth_sha256=sha(truthpath))
    assert all(sha(p)==v for p,v in pins.items())
    result=dict(passed=True,model_sha256=sha(HERE/'model.py'),protocol_sha256=sha(HERE/'PROTOCOL.md'),
                source_bindings=pins,checks=checks,utc=datetime.now(timezone.utc).isoformat(),
                scientific_gain_claim=False,training_started=False,
                truth_scope='Existing1e9 means of the two fixed TRAIN development nets only')
    write(HERE/'ready_v1.json',result)
    print(json.dumps(dict(passed=True,checks=checks)),flush=True)


if __name__=='__main__':
    main()
