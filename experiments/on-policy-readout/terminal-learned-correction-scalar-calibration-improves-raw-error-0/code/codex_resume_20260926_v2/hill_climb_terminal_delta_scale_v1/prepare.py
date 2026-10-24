"""Prepare a current-native terminal-delta scalar study without numerical execution."""
from pathlib import Path
import hashlib,json,ast
D=Path(__file__).resolve().parent;H=D.parent;P=H/'hill_climb_pair_rr_inverse_v1'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,v):
 with Path(p).open('x') as f:f.write(v if isinstance(v,str) else json.dumps(v,indent=2,allow_nan=False)+'\n')
source=pin(P/'candidate.py');assert source['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
original=H/'hill_climb_e1_ynt_psdb_d5_v1/native_probe.py';s=original.read_text()
before='est=mod.Estimator()\n'
after='''capture = {}
original_scalar = mod.nonlinear_scalar
original_gnn = mod.gnn_step
def scalar_capture(*args, **kwargs):
    value = original_scalar(*args, **kwargs)
    capture['physical'] = value[0]
    capture['scalar_calls'] = capture.get('scalar_calls', 0) + 1
    return value
def gnn_capture(layer, *args, **kwargs):
    value = original_gnn(layer, *args, **kwargs)
    if layer == 15:
        capture['delta'] = value[0]
        capture['terminal_calls'] = capture.get('terminal_calls', 0) + 1
    return value
mod.nonlinear_scalar = scalar_capture
mod.gnn_step = gnn_capture
est=mod.Estimator()
'''
assert s.count(before)==1;s=s.replace(before,after,1)
before='        t0=time.monotonic();c0=time.process_time()\n'
after='        capture.clear()\n'+before
assert s.count(before)==1;s=s.replace(before,after,1)
before="        flops_used=int(ctx.summary_dict()['flops_used'])\n"
after=before+'''        assert capture['scalar_calls'] == capture['terminal_calls'] == 1
        physical = np.asarray(capture['physical'], dtype=np.float64).copy()
        delta = np.asarray(capture['delta'], dtype=np.float64).copy()
        assert physical.shape == delta.shape == (1024,)
        assert np.isfinite(physical).all() and np.isfinite(delta).all()
        replay = np.maximum(physical + delta, 0.0).astype(np.float32).astype(np.float64)
        assert replay.tobytes() == y[15].tobytes(), 'scale1 terminal replay'
        np.savez(out/f'terminal_{rank}.npz', physical=physical, delta=delta)
        capture.clear()
'''
assert s.count(before)==1;s=s.replace(before,after,1)
ast.parse(s);compile(s,str(D/'capture.py'),'exec');once(D/'capture.py',s)
cfg=json.loads((P/'native_screen_continuation.json').read_text())
base=cfg['arms']['candidate'];assert base['source']==source
once(D/'protocol.json',dict(schema='current-native-terminal-delta-one-scalar-v1',source=source,retained_parent=pin(P/'retained_component.json'),official_bank=pin(H/'official_bank_S35_v1.json'),capture_driver=pin(D/'capture.py'),original_driver=pin(original),capture_driver_parent_inverse='Only retained-array wrappers and post-budget capture/replay assertions are added; estimator source is unchanged.',base_blocks=base['blocks'],fit_ranks=[35,100],check_ranks=[36,45,52,58,64,70],method='Let p=physical+existing_delta, d=existing_delta. Fit one shared scalar s=1-sum((p-truth)*d)/sum(d*d), using only units where p>0 on the two fit networks. Freeze s once. Evaluate all units as maximum(physical+s*delta,0).astype(float32).astype(float64), matching actual return order.',fit_gates=dict(min_raw_gain_pct=0.1,min_wins=1,max_regression_pct=1.0,positive_finite_scale=True),check_gates=dict(min_raw_gain_pct=0.1,min_wins=4,max_regression_pct=1.0,bootstrap_raw_ratio_upper_less_than_one=True,bootstrap_resamples=50000,bootstrap_seed=20260926),integrity=['Current RRPI source SHA and baseline source/roster pins match.','All16 baseline prediction FILE bytes and complete metered bill match captures at the same two/six-network order.','At scale1 terminal replay bytes match the actual final output.','Capture retains only two1024-element values and converts them after BudgetContext.','Separate check networks have been historically reused; they are not pristine holdout data.'],limitations=['Training fit is a one-step active-support least-squares proposal, not an exact optimum of the clipped/rounded objective.','Offline gains are not native candidate or serving qualification.','No alternate feature, subset, scalar refit, clipping window, or layer variant after a failed fixed gate.','Source-dependent _lr_factor and every other estimator function remain unchanged.'],authority='Root serial guard only. Fit capture2 first; capture6 only after FIT2 gates pass. Then a frozen scalar candidate needs actual native score and resource qualification before retention; official banking still requires >=1% versus S35 on the fixed40 and own serving checks.',resource_limits=dict(capture2_seconds=150,capture6_seconds=300,assessment_seconds=120,rss_bytes=7500000000,added_swap_bytes=0),model_executed=False))
once(D/'preparation.json',dict(builder=pin(__file__),protocol=pin(D/'protocol.json'),driver=pin(D/'capture.py'),source=source,source_unchanged=True,stdlib_only=True))
print(json.dumps({'protocol':pin(D/'protocol.json'),'preparation':pin(D/'preparation.json'),'driver':pin(D/'capture.py')}))
