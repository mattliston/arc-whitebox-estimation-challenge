"""Source-only prospective native three-dial acquisition; never imports a model."""
from pathlib import Path
import ast, hashlib, json, shlex

D=Path(__file__).resolve().parent; H=D.parent; P=H.parent
R=P.parents[3]
BASE=H/'hill_climb_s34_pair_read7_scan_v1/candidate.py'
NATIVE=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
LIGHT=R/'.venv/bin/python'
GUARD=P/'codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py'
DRIVER=H/'hill_climb_e1_ynt_psdb_d5_v1/native_probe.py'
FIT=[35,100,36,45]; CHECK=[52,58,64,70,76,80,84,88]
VALUES=dict(bscale=1.0383394097164294,newborn=0.9932375877606482,c1=0.5026934507950427)

def pin(path):
    p=Path(path).resolve(); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)

def once(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')

def suffix(values):
    lines=['\n# Prospective native joint3 literal calibration; physical semantics unchanged.']
    if 'bscale' in values:lines.append('_S19_BSCALE = '+repr(values['bscale']))
    if 'newborn' in values:lines.append('_S19_LAM = (_S19_LAM[0], '+repr(values['newborn'])+')')
    if 'c1' in values:lines.append('_K31R_TABLE = {l: (c[0], '+repr(values['c1'])+') for l, c in _K31R_TABLE.items()}')
    return '\n'.join(lines)+'\n'

assert pin(BASE)['sha256']=='2212448b123d4be2dadf9c4c8642b49ad1cb3e073a1cc49684e4fc083a19c14f'
assert LIGHT.exists() and NATIVE.exists()
source=BASE.read_bytes(); tree=ast.parse(source)
assert b'_S19_GN_TORCH = False' in source
assert b'_S19_BSCALE = 1.0383394097164294' in source
assert b'_S19_LAM = (1.0032287195131746, 0.9932375877606482)' in source
assert b'_K31R_TABLE = {l: (0.5341766552087044, 0.5026934507950427)' in source
roles=P/'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json'
part=json.loads(roles.read_text())
assert set(FIT+CHECK)<=set(part['train']) and not set(FIT+CHECK)&set(part['valid']) and not set(FIT)&set(CHECK)
store=P/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
sm=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
arms=[]
for dial in VALUES:
    for direction,mult in [('minus',.98),('plus',1.02)]:
        name=dial+'_'+direction; p=D/(name+'.py'); tail=suffix({dial:VALUES[dial]*mult}).encode()
        with p.open('xb') as f:f.write(source+tail)
        assert p.read_bytes()[:-len(tail)]==source
        assert len(ast.parse(p.read_bytes()).body)==len(tree.body)+1
        tag=D.name+'_'+name+'_fit4_v1'
        command=[str(NATIVE),'-B',str(DRIVER),str(p),','.join(map(str,FIT)),tag]
        record=D/(name+'_guard.json')
        arms.append(dict(name=name,dial=dial,multiplier=mult,source=pin(p),appended_source=tail.decode(),
            tag=tag,output=str(H/tag),guard=str(record),limits=dict(seconds=240,rss_bytes=7500000000),command=command,
            launch=[str(LIGHT),'-B',str(GUARD),'--record',str(record),'--seconds','240','--rss-limit','7500000000',*command]))
controls=[]
for size,ranks in [(2,[35,100]),(6,[36,45,52,58,64,70])]:
    folder=H/f'hill_climb_s34_pair_read7_scan_v1_train{size}_v1'
    controls.append(dict(ranks=ranks,manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),
        guard=pin(BASE.parent/f'native{size}_guard.json'),
        predictions={str(r):pin(folder/f'pred_{r}.npy') for r in ranks if r in FIT}))
data=[]
for r in FIT:
    rm=store/sm[r]['path']; row=json.loads(rm.read_text()); w=rm.parent/'weights.npy'
    assert row['role_rank']==r and pin(w)['sha256']==row['files']['weights.npy']['sha256']
    data.append(dict(rank=r,row_manifest=pin(rm),weights=pin(w),truth=pin(P/'fable_refit_v1/truth688'/f'truthpost_rank{r:04d}.npz')))
protocol=dict(schema='native-joint3-one-step-v1',root_admitted=True,fit_ranks=FIT,check_ranks=CHECK,
    root_selection=pin(H/'s34_native_three_dial_root_selection_v1.json'),
    baseline=pin(BASE),baseline_native8=pin(BASE.parent/'native8_assessment.json'),controls=controls,data=data,
    roles=pin(roles),store_manifest=pin(store/'manifest.json'),baseline_values=VALUES,dial_order=list(VALUES),arms=arms,
    driver=pin(DRIVER),guard_source=pin(GUARD),builder=pin(__file__),fit_source=pin(D/'fit.py'),runtime=str(NATIVE),
    setup=dict(seconds_max=4.1,counts=dict(bound=687,lvfuse=649,psdb_fused=9,pair_block_indices=1,read7_fused=4)),
    step_fraction=.02,ridge_diagonal_fraction=.1,box=[-1.,1.],solver='27 active sets; fixed ridge; zero columns fixed at0',
    objective='Final-layer pooled MSE weighted per network by saved baseline max(.1,util); measured arm bill changes diagnostic only.',
    gates=dict(predicted_fit_gain_pct_min=.5,actual_fit_gain_pct_min=.1,actual_fit_wins_min=3,
        check_gain_pct_min=.1,check_wins_min=6,check_ci_upper_below=0.,broad_launch_total_gain_vs_s34_pct_min=.8,
        final_valid40_bank_gain_pct_min=1.),parity_layers=[0],one_endpoint_only=True,
    continuation='One actual FIT4 endpoint then disjoint CHECK8, normal own boundary/VALID40/full-serving qualification. FIT ranks never independent validation. Keep smaller confirmed component for later combination; no ridge, step, direction or endpoint search.',
    uncertainty='Symmetric even remainders are diagnostics. Native differences need not be exact derivatives across gating/QR/rounding changes.',
    semantics='b after bfix before trace; newborn signed lambda only, carry unchanged; c1 shared pair/GN remainder with GN_TORCH=False, c0 and TABLE_GN unchanged.',
    history=[pin(H/'teammate_native_joint_calibration_audit_v1/RESULT.md'),pin(P/'fable_covhunter_v1/gn_joint_fit.py'),
        pin(P/'fable_covhunter_v1/s6bcard_joint2.py'),pin(P/'fable_physicist_v1/refit6b/launch_loopR8s.sh')],
    observations_read=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
fit_command=[str(NATIVE),'-B',str(D/'fit.py'),pin(D/'protocol.json')['sha256']]
commands=dict(acquisition=[dict(name=a['name'],argv=a['launch']) for a in arms],
    fit=[str(LIGHT),'-B',str(GUARD),'--record',str(D/'fit_guard.json'),'--seconds','60','--rss-limit','1000000000',*fit_command],
    execution='Root only, serial; acquisition then one fit. No model runs by builder.')
once(D/'commands.json',commands)
with (D/'commands.txt').open('x') as f:
    for a in commands['acquisition']:f.write(shlex.join(a['argv'])+'\n\n')
    f.write(shlex.join(commands['fit'])+'\n')
once(D/'build.json',dict(parent=pin(BASE),builder=pin(__file__),fit=pin(D/'fit.py'),protocol=pin(D/'protocol.json'),
    candidates=[a['source'] for a in arms],exact_parent_prefix=True,only_one_appended_assignment_per_arm=True,
    commands=pin(D/'commands.json'),model_executed=False))
print(json.dumps({f:pin(D/f) for f in ['protocol.json','build.json','commands.json','fit.py']}))
