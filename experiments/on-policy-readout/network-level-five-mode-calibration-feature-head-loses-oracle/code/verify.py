"""Independent scalar score reconciliation and saved-model replay."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

D=Path(__file__).resolve().parent;G=D.parent/'native_final_graph_head_v1';P2=D.parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((D/'protocol.json').read_text())
assessment=json.loads((D/'assessment.json').read_text())
records={r['rank']:r for r in json.loads((G/'capture_complete.json').read_text())['rows']}
reported={r['rank']:r for r in assessment['per_network']}
largest=0.;replay=0.;checked=0;max_oracle_normal_residual=0.
for fold in range(4):
    folder=D/f'fold_{fold}'
    result=json.loads((folder/'result.json').read_text())
    model={}
    predictions={}
    for mode in ('intercept','network'):
        with np.load(folder/f'{mode}_model.npz') as saved:model[mode]={k:saved[k] for k in saved.files}
        predictions[mode]=np.load(folder/f'{mode}_prediction.npy')
    for j,rank in enumerate(result['test_ranks']):
        assert rank in cfg['ranks'] and rank not in cfg['excluded_check32']+cfg['excluded_valid40']
        packet=G/records[rank]['path'];assert sha(packet)==records[rank]['sha256']
        with np.load(packet) as saved:
            x=saved['control'].astype(float);s=saved['sigma'].astype(float);base=saved['prediction'][15].astype(float)
        with np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz') as saved:truth=saved['mc_post'][15].astype(float)
        values={'bank':base}
        a=x[:,3];g=x[:,5];pdf=x[:,6]
        physical=np.column_stack((g,a*g,pdf,a*pdf,(a*a-1)*pdf))*s[:,None]*.001
        for mode in model:
            m=model[mode]
            z=np.r_[1.,np.clip((x.mean(0)-m['mean'])/m['scale'],-8,8)]
            if mode=='intercept':z=z[:1]
            # Fold the output whitening into the five physical coefficients;
            # do not reuse the fitter's batched einsum prediction path.
            physical_coeff=m['inverse']@(z@m['coefficient'])
            delta=sum(physical[:,k]*physical_coeff[k] for k in range(5))
            expected=np.maximum(base+delta,0).astype(np.float32).astype(float)
            difference=float(np.max(np.abs(expected-predictions[mode][j])))
            replay=max(replay,difference)
            assert difference<=2e-8
            values[mode]=predictions[mode][j]
        coeff=np.linalg.solve(physical.T@physical,physical.T@(truth-base))
        values['oracle']=base+physical@coeff
        gradient=physical.T@(values['oracle']-truth)
        max_oracle_normal_residual=max(max_oracle_normal_residual,float(np.linalg.norm(gradient)))
        for key,value in values.items():
            score=math.fsum((float(v)-float(t))**2 for v,t in zip(value,truth))/1024
            error=abs(score-reported[rank][key])/max(score,1e-30)
            largest=max(largest,error)
            assert error<1e-10,(rank,key,error)
            checked+=1
        # Synthetic aggregate reconstruction is already separate; here check
        # each actual least-squares oracle is no worse than the bank.
        assert reported[rank]['oracle']<=reported[rank]['bank']*(1+1e-12)
result=dict(passed=True,networks=128,independent_scores=checked,
    max_relative_score_disagreement=largest,max_model_replay_absolute_error=replay,
    max_oracle_normal_residual=max_oracle_normal_residual,
    assessment_sha256=sha(D/'assessment.json'),verify_sha256=sha(Path(__file__)))
with (D/'verification.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result),flush=True)
