"""Source/stdlib-only fixed response64 study; never imports the estimator."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
H = D.parent
P = H.parent
R = P.parents[3]
BASE = H / 'hill_climb_prefix128_fusion_v1'
T = H / 'hill_climb_hidden_mean_projection_v1'
SOURCE_SHA = 'f9f4790b3c65f730404425e8923c767dc2f895fdcc1e14f14cc649286ad3b69f'

def pin(p):
    p = Path(p); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))

def read(p):
    return json.loads(Path(p).read_text())

def write(name, value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2) + '\n'
    with (D / name).open('x') as f:
        f.write(text)

parent = BASE / 'candidate.py'
assert pin(parent)['sha256'] == SOURCE_SHA
before = '            response_indices, response_mask = select(response_score, gate_floor)\n'
after = '            response_indices, response_mask = _response64_prejoin(l, response_score, gate_floor, live_indices, lam, shared_old)\n'
original = parent.read_text()
suffix = (D / 'fragment.py').read_text()
assert original.count(before) == 1 and '_response64_prejoin' not in original
source = original.replace(before, after) + suffix
assert source[:-len(suffix)].replace(after, before) == original
old_ast, new_ast = ast.parse(original), ast.parse(source)
assert len(new_ast.body) == len(old_ast.body) + 1
changed = [(a.name, a.lineno) for a, b in zip(old_ast.body, new_ast.body)
           if ast.dump(a) != ast.dump(b)]
assert changed == [('_predict_binary', 3214)]
assert new_ast.body[-1].name == '_response64_prejoin'
compile(source, str(D / 'candidate.py'), 'exec')
write('candidate.py', source)
write('source.diff', ''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True), fromfile=str(parent), tofile=str(D / 'candidate.py'))))

gates = dict(adjusted_pct_at_most=-0.1, strict_adjusted=False, wins_min=1,
             ci_upper_below=None, max_per_network_regression_pct=1.0,
             additional6_gain_required=False)
protocol = dict(
    schema='response64-prejoin-native2-v1', parent=pin(parent), candidate=pin(D/'candidate.py'),
    source_scope='Only the active response selection call and one appended helper; fixed L4-6, no old columns, fill next64 only up to live count and only identical non-None multi-chunk KS plan. READ7 eligibility excluded on both shapes.',
    selection='Root-selected metadata-aware accuracy trial; no rank IDs, host tensor values, floor grid or additional windows. Original select already fills32; this changes response only to64 when admitted.',
    invariants=['Original live selection and births selection unchanged.', 'Stable argsort of identical response_score retains original prefix and adds highest remaining alive rows.', 'Returned response mask flows to omitted and _R320_RESP before the existing push/read.', 'Fixed prejoin scope has no shared-old read; CP diagonal and original omitted-radial algorithm remain unchanged.', 'No new cache, arena or setup binding; _ks_plan keeps only its existing public-shape metadata.'],
    numerical_caveat='Approximate accuracy change. Added rows alter packed arithmetic, omission repair, downstream means and later shapes. Equal initial kernel geometry is not a full-network bill or byte-parity promise.',
    direct_cost_forecast=dict(saved_shape_scope='Historical TRAIN35/100 initial geometry only', rank35_flops_delta=-16904192, rank100_flops_delta=-7479296, mean_flops_delta=-12191744, excludes='small extra argsort/mask bookkeeping, and changed downstream trajectories'),
    closest_prior=dict(response32_already_active=True, response0125_raw_regression_pct=2.2229868691151733, exact64_match_found=False, search_scope='Named candidate paths and RESULTS plus analyst/physicist/engineer2/porter result records; not exhaustive history'),
    evidence=[pin(H/'young_read_shapes_train2_v1/shape_census.json'), pin(H/'hill_climb_s34_responsefloor0125_v1/protocol.json'), pin(H/'hill_climb_s34_responsefloor0125_v1/native2_assessment.json')],
    native2_gates=gates, ranks=[35,100], parity_layers=[0], finite_all16_required=True,
    limits=dict(seconds=150, rss_bytes=7500000000, zero_added_swap=True),
    stage_authority='NATIVE2_ONLY; no continuation, boundary, broad or submission authorization from this config', model_executed=False)
write('protocol.json', protocol)
write('build.json', dict(candidate=pin(D/'candidate.py'), protocol=pin(D/'protocol.json'), builder=pin(__file__), fragment=pin(D/'fragment.py'),
    source_inverse=dict(parent=pin(parent), suffix=suffix, edits=[dict(before=before, after=after)], exact_inverse=True),
    only_changed_function=changed, added_function='_response64_prejoin', ast_compile=True, source_diff=pin(D/'source.diff')))

cfg = copy.deepcopy(read(T/'native_screen.json'))
cfg['provenance'] = [pin(D/'build.json'), pin(D/'protocol.json'), pin(T/'native_screen.json'),
                     pin(H/'hill_climb_prefix128_edge_bias_hoist_v1/engine_inverse.json'),
                     pin(BASE/'native_screen_continuation.json'), pin(BASE/'build.json')]
cfg['arms']['candidate']['source'] = pin(D/'candidate.py')
for size, block in cfg['arms']['candidate']['blocks'].items():
    block['tag'] = D.name + '_train' + size + '_v1'
    block['guard'] = str(D / ('native' + size + '_guard.json'))
for size, stage in cfg['stages'].items():
    stage['assessment_guard'] = str(D / ('native' + size + '_assess_guard.json'))
    stage['result'] = str(D / ('native' + size + '_assessment.json'))
cfg['stages']['2']['gates'] = gates
cfg['boundary_before8']['receipt'] = str(D/'NATIVE8_NOT_AUTHORIZED.json')
cfg['boundary_before8']['assessment_guard'] = str(D/'NATIVE8_NOT_AUTHORIZED_guard.json')
cfg['interpretation'] = 'Fixed response64 prejoin accuracy pilot against actual P128 TRAIN2 controls. Standard approximate gates unchanged; N2-only.'
assert cfg['arms']['base']['source'] == pin(parent) == cfg['arms']['parity_parent']['source']
assert cfg['parity_layers'] == [0] and cfg['no_further_stage_authority'] is True
write('native_screen.json', cfg)

template = read(T/'commands_native2.json')
native = template['commands']['native2'][:]
native[-3:] = [str(D/'candidate.py'), '35,100', D.name+'_train2_v1']
assess = template['commands']['native2_assessment'][:]
assess[-3:] = [str(D/'native_screen.json'), pin(D/'native_screen.json')['sha256'], '2']
commands = dict(native2=native, native2_assessment=assess)
guards = dict(native2=D/'native2_guard.json', native2_assessment=D/'native2_assess_guard.json')
guarded = {}
for name, command in commands.items():
    prefix = template['guarded_commands'][name][:10]
    assert prefix[3] == '--record' and prefix[5] == '--seconds' and prefix[7] == '--rss-limit'
    prefix[4] = str(guards[name])
    # The command starts at index9; keep the original LIGHT guard prefix only.
    guarded[name] = prefix[:9] + command
write('commands_native2.json', dict(config=pin(D/'native_screen.json'), commands=commands, guarded_commands=guarded,
    environment={key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
    stdout_log=str(D/'native2.log'), native2_guard=str(guards['native2']), assessment_guard=str(guards['native2_assessment']),
    guard=template['guard'], scope='Root-only guarded N2 and its assessment; no other model commands.', executed=False))

seen = set()
def check(value):
    if isinstance(value, dict):
        if all(k in value for k in ('path','sha256','bytes')):
            assert pin(value['path']) == value
            seen.add(value['path'])
        for x in value.values(): check(x)
    elif isinstance(value, list):
        for x in value: check(x)
for name in ('build.json','protocol.json','native_screen.json','commands_native2.json'):
    check(read(D/name))
print(json.dumps(dict(candidate=pin(D/'candidate.py'), config=pin(D/'native_screen.json'), commands=pin(D/'commands_native2.json'), checked_pins=len(seen))))
