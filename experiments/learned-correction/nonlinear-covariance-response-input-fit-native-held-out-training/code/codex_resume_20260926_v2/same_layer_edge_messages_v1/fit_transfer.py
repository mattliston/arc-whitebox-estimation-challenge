"""Frozen two-FIT forward-response diagnostic; no optimization or CHECK access."""
from pathlib import Path
import hashlib
import json
import time
import model
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((HERE/'fit_transfer_protocol.json').read_text())
assert protocol['ranks']==model.PROTOCOL['fit_ranks'][:2]
assert not (HERE/'fit_transfer.json').exists()
api=model.initialize();t=api['torch'];np=api['np']
assert api['binding']==json.loads((HERE/'qualification.json').read_text())['binding']
records=[];started=time.monotonic()
for mode in protocol['arms']:
    g=json.loads((HERE/f'transfer_native_{mode}_guard.json').read_text())
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    result=json.loads((HERE/f'fit_{mode}'/'result.json').read_text())
    payload=HERE/f'fit_{mode}'/'terminal.npz';assert sha(payload)==result['terminal_sha256']
    with np.load(payload) as z:
        theta=t.from_numpy(z['theta'].copy());mean=t.from_numpy(z['mean'].copy());scale=t.from_numpy(z['scale'].copy())
    native_dir=H2/f'same_layer_edge_{mode}_fit2_transfer_v1'
    nm=json.loads((native_dir/'manifest.json').read_text())
    assert nm['sha256']==sha(Path(nm['candidate']))==json.loads((HERE/f'build_{mode}_terminal.json').read_text())['sha256']
    for rank in protocol['ranks']:
        with np.load(HERE/'torch_baselines.npz') as z:bt=z[str(rank)][15].copy()
        bn=np.load(H2/'response_message_capture_train12_v1'/f'pred_{rank}.npy')[15]
        yn=np.load(native_dir/f'pred_{rank}.npy')[15]
        truth=np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
        with t.no_grad():
            w=api['weights'](rank);cd=api['card'](w,theta,mode,mean,scale);pred=api['run'](cd)
            yt=pred[-1].numpy().copy()
        dt=yt-bt;dn=yn-bn;anchor=bn+dt
        assert np.linalg.norm(dt)>0 and np.linalg.norm(dn)>0
        baseline_mse=float(np.mean((bn-truth)**2));tmse=float(np.mean((anchor-truth)**2));nmse=float(np.mean((yn-truth)**2))
        row=dict(mode=mode,rank=rank,baseline_mse=baseline_mse,anchored_torch_mse=tmse,native_mse=nmse,
            anchored_torch_pct=100*(tmse/baseline_mse-1),native_pct=100*(nmse/baseline_mse-1),
            delta_cosine=float(np.dot(dt,dn)/(np.linalg.norm(dt)*np.linalg.norm(dn))),
            delta_relative_l2_discrepancy=float(np.linalg.norm(dt-dn)/np.linalg.norm(dn)),
            torch_delta_rms=float(np.sqrt(np.mean(dt*dt))),native_delta_rms=float(np.sqrt(np.mean(dn*dn))))
        records.append(row);print(json.dumps(row),flush=True)
        del w,cd,pred;model.collect()
report=dict(records=records,fit_ranks=protocol['ranks'],protocol_sha256=sha(HERE/'fit_transfer_protocol.json'),
    script_sha256=sha(HERE/'fit_transfer.py'),elapsed_s=time.monotonic()-started,optimizer_steps=0,
    check_or_valid_accessed=False,pilot_gates_changed=False,
    interpretation='Two previously used FIT networks only; measures frozen forward-response transfer, not a certified derivative or held-out generalization.')
with (HERE/'fit_transfer.json').open('x') as fp:json.dump(report,fp,indent=2,allow_nan=False)
