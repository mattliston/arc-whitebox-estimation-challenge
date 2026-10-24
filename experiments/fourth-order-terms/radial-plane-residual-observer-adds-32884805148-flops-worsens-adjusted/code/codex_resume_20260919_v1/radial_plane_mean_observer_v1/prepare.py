import ast
from common import *

assert sha(PARENT) == PARENT_SHA
previous = D / 'continuation_checkpoint_v77.json'
assert sha(previous) == '1cab0c658cf51803d22ca8e0a62ec03ef665ef4b9d0608ed4148489b6847ca01'
for key in ['study', 'official_bank']:
    check(read(previous)[key])
source = PARENT.read_text()
tree = ast.parse(source)
inverses, replacements = [], []
for name in ['_nonlinear_first_gaussian', '_nonlinear_pair_batch', '_predict_binary']:
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
    original = ast.get_source_segment(source, node)
    if name != '_predict_binary':
        edits = [('    return (m, cov, post21, radial, aux)',
                  "    aux['r22_post22'] = post22\n    return (m, cov, post21, radial, aux)", 1)]
    else:
        edits = [
            ('        shared_old = _shared_old384_state()',
             "        shared_old = _shared_old384_state()\n        r22_state = {'mean': None, 'plane': None}", 1),
            ('                m, aux = nonlinear_scalar(mu, v, k3, radial * v * v)',
             '                m, aux = nonlinear_scalar(mu, v, k3, radial * v * v)\n                if _R22_ENABLED:\n                    _r22_advance(r22_state, w32, w, aux)', 1),
            ('            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)',
             "            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n            if _R22_ENABLED:\n                _r22_advance(r22_state, w32, w, aux)\n                if l < depth - 1:\n                    r22_state['plane'] = _r22_source(C, aux['r22_post22'])", 1),
            ('means.append(m.copy())', 'means.append(_r22_read(m, r22_state))', 2)]
    changed = original
    for before, after, count in edits:
        assert changed.count(before) == count, (name, before)
        changed = changed.replace(before, after)
    recovered = changed
    for before, after, count in reversed(edits):
        assert recovered.count(after) == count
        recovered = recovered.replace(after, before)
    assert recovered == original
    replacements.append(changed)
    inverses.append(dict(function=name, edits=edits,
        original_sha256=__import__('hashlib').sha256(original.encode()).hexdigest(),
        appended_sha256=__import__('hashlib').sha256(changed.encode()).hexdigest()))
fragment = ((HERE / 'helper.py').read_text() + '\n\n' + '\n\n'.join(replacements[:2])
            + '\n\nnonlinear = _nonlinear_pair_batch\n\n' + replacements[2] + '\n').encode()
with (HERE / 'fragment.py').open('xb') as f:
    f.write(fragment)
candidate = PARENT.read_bytes() + b'\n\n' + fragment
ast.parse(candidate.decode())
with (HERE / 'candidate.py').open('xb') as f:
    f.write(candidate)
write(HERE / 'build.json', dict(parent=pin(PARENT), candidate=pin(HERE / 'candidate.py'),
    helper=pin(HERE / 'helper.py'), fragment=pin(HERE / 'fragment.py'), inverses=inverses,
    whole_parent_prefix_exact=candidate[:-len(fragment)-2] == PARENT.read_bytes(),
    changes_parent_trajectory=False, previous_checkpoint=pin(previous)))
old_inputs = D / 'full_cubic_support_native_v1/inputs.json'
inputs = read(old_inputs)
for row in inputs['rows']:
    for key in ['weights', 'truth', 'parent_result', 'prediction']:
        check(row[key])
contexts = {}
for key, name in [('triple', 'parent_triple'), ('cold100', 'parent_cold100')]:
    p = D / f'full_lower_multilayer_mean_observer_v1/{name}/result.json'
    contexts[key] = dict(pin=pin(p), records=read(p)['records'])
inputs['matching_contexts'] = contexts
inputs['copied_row_metadata'] = pin(old_inputs)
write(HERE / 'inputs.json', inputs)
external = [PARENT, previous, D / 'official_best_331684.json', D / 'run_guarded.py', old_inputs,
    D / 'complete_catalog_depth_v1/closeout.json',
    D / 'full_lower_multilayer_mean_observer_v1/RESULT.md',
    D / 'terminal_tc_information_v1/SOURCE_REVIEW.md',
    P2 / 'public_g29_bond_kernel_v1/k22_chain.py',
    P2 / 'public_g29_bond_kernel_v1/README.md',
    A / 'GAUGE_RADIAL_PROJECTION_INTERPRETATION_V1.md',
    D / 'unit_norm_gauge_v1/PROTOCOL.md']
external += [Path(v['pin']['path']) for v in contexts.values()]
for p in HERE.glob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
write(HERE / 'sources_sealed.json', dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]
    + [pin(p) for p in external], targets_not_decoded=True))
print(dict(candidate=pin(HERE / 'candidate.py'), function_inverses=len(inverses), sealed=True), flush=True)

