from pathlib import Path
import ast, copy, difflib, hashlib, json, datetime
H=Path(__file__).resolve().parent.parent
D=Path(__file__).resolve().parent
def pin(p):
 p=Path(p).resolve(); b=p.read_bytes()
 return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def save(n,v): 
 with (D/n).open('x') as f: json.dump(v,f,indent=2);f.write('\n')
parent=H/'hill_climb_pair_rr_inverse_v1/candidate.py'
assert pin(parent)['sha256']=='a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'
fragment='''# Algebraically identical signed-cohort cubic diagonal in stored FP32 legs.
def _cp_factored_diagonal(pp, lam, state):
    hold = _E2Hold()
    plus, minus = [], []
    cursor = 0
    for start, width in state['cohorts']:
        start, width = int(start), int(width)
        assert start == cursor and width >= 0 and width % 2 == 0
        half = width // 2
        plus.extend(range(start, start + half))
        minus.extend(range(start + half, start + width))
        cursor += width
    assert cursor == int(pp.shape[1]) == int(lam.shape[0])
    ip = hold(fnp.asarray(plus, dtype=fnp.int64))
    im = hold(fnp.asarray(minus, dtype=fnp.int64))
    t = hold(fnp.take(pp, ip, axis=1))
    s = hold(fnp.take(pp, im, axis=1))
    delta = hold(t - s)
    square = hold(t * t)
    hold(fnp.add(square, hold(t * s), out=square))
    hold(fnp.add(square, hold(s * s), out=square))
    hold(fnp.multiply(square, delta, out=square))
    hold(fnp.multiply(square, hold(fnp.take(lam, ip)), out=square))
    return hold(fnp.sum(square, axis=1)).astype(fnp.float64)
'''
(D/'fragment.py').write_text(fragment)
original=parent.read_text()
old='''                _e2_t3 = _e2_pp2 * pp
                _e2_t3h = fnp.multiply(_e2_t3, lam, out=_e2_t3)   # E2-M: in place (same products and order): one (n,K) transient fewer at the live peak
                diagonal = _e2_H(fnp.sum(_e2_t3, axis=1)).astype(fnp.float64)
                diagonal = _ott_diagonal(diagonal, lam)
                del _e2_pp2, _e2_t3, _e2_t3h'''
new='''                diagonal = _cp_factored_diagonal(pp, lam, shared_old)
                diagonal = _ott_diagonal(diagonal, lam)
                del _e2_pp2'''
old2="    k3_rows = fnp.sum(_cp_diagonal_cube(pp_rows) * lam, axis=1).astype(fnp.float64)   # F32 (E-29)"
new2="    k3_rows = _cp_factored_diagonal(pp_rows, lam, state)"
assert original.count(old)==original.count(old2)==1
candidate=original.replace(old,new).replace(old2,new2)+'\n\n'+fragment
assert candidate.removesuffix('\n\n'+fragment).replace(new,old).replace(new2,old2)==original
ast.parse(candidate)
(D/'candidate.py').write_text(candidate)
(D/'source.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile=str(parent),tofile=str(D/'candidate.py'))))
save('source_delta.json',dict(parent=pin(parent),candidate=pin(D/'candidate.py'),fragment=pin(D/'fragment.py'),whole_source_inverse=True,changed_functions=['_predict_binary active implementation','_l15_rowcut_k3'],added_function='_cp_factored_diagonal'))
protocol=dict(created=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='One fixed native2 accuracy trial from RRPI; no fit, sweep or parameter change.',hypothesis='Avoid subtracting independently rounded weighted FP32 cubes by factoring each signed cohort pair before FP32 reduction. Same real moment of actual stored legs; does not recover lost upstream factor precision.',formula='sum_pairs lambda_plus*(t-s)*(t*t+t*s+s*s)',coefficient_invariant='Each cohort stores [a,-a] from same FP32 a. Every carry scale is identical across both halves. Metadata preserves complete original signed roster.',dedup='CPsum64 only changed reduction; old polarization changed plane read while preserving scalar diagonal; stored newborn diagonal changed postbirth target. All failed and remain closed.',source=pin(D/'candidate.py'),parent=pin(parent),primitive_gate='Same actual helper extracted, exact zero pairs and exact small-integer fixtures; normalized error <=2e-6 and cancellation-heavy RMS error <=0.25*old against FP64 reference.',pricing='Hidden old extra two full products+sum=3K; new factor eight half products/adds plus sum=4.5K, extra1.5K scalar ops plus paid gathers. Under conservative K<=30720 and n<=1024, 15*1.5*n*K <708M FLOPs (<0.27% of263B), before gather charges. Actual complete bill decisive.',gates=dict(native2_raw_pct_max=-.1,native2_adjusted_pct_max=-.1,wins_min=1,max_regression_pct=1.,native8_raw_pct_max=-.1,native8_adjusted_pct_max=-.1,wins_min8=6,ci_upper_max=0.,additional6_gain_required=True,bank40_pct_max=-1.),parity_layers=[0],further_stages='Native2 only admitted after primitive PASS. If N2 passes, own boundary then additional6. Full40 only if combined projected>=1% vsS35. Failure closes this fixed scalar factored trial without precision/layer/strength variants.')
save('protocol.json',protocol)
engine=H/'hill_climb_rrpi_tail_warm2_v1/screen_accuracy.py'
(D/'screen_accuracy.py').write_bytes(engine.read_bytes())
config=json.loads((H/'hill_climb_old_tail_diagonal_hermite_v1/native_screen.json').read_text())
config['engine']=pin(D/'screen_accuracy.py')
config['provenance']=[pin(D/'protocol.json'),pin(D/'source_delta.json'),pin(D/'fragment.py'),pin(parent),pin(H/'hill_climb_pair_rr_inverse_v1/retained_component.json')]
arm=config['arms']['candidate'];arm['source']=pin(D/'candidate.py')
for n,b in arm['blocks'].items():
 b['tag']=D.name+'_train'+n+'_v1';b['guard']=str(D/('native'+n+'_guard.json'))
 b.pop('artifacts',None)
for n,st in config['stages'].items():
 st['result']=str(D/('native'+n+'_assessment.json'));st['assessment_guard']=str(D/('native'+n+'_assess_guard.json'))
config['parity_layers']=[0]
config['boundary_before8']['receipt']=str(D/'NATIVE8_NOT_AUTHORIZED.json')
config['boundary_before8']['assessment_guard']=str(D/'NATIVE8_NOT_AUTHORIZED_guard.json')
config['interpretation']='One fixed factored signed-cohort cubic scalar diagonal; all learned parameters and plane machinery intact. Actual RRPI saved controls. Native2 only until own boundary.'
save('native_screen.json',config)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'),protocol=pin(D/'protocol.json'))))

