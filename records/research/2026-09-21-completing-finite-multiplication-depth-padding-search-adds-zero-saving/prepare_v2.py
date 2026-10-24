import ast
from common import *
OLD = D / 'full_lower_cost_frontier_v1'
assert sha(D / 'continuation_checkpoint_v76.json') == 'cce56409d9103cf2a57cc75e99644d7717bfb2fcdaefda3d37fd23cdb7e1f9a0'
previous = read(D / 'continuation_checkpoint_v76.json')
check(previous['study']); check(previous['official_bank'])
old_closeout = read(OLD / 'closeout.json')
for p in old_closeout['artifacts'] + old_closeout['external_pins']:
    check(p)
changes = {
    'reprice.py': [('DEPTH = 5', 'DEPTH = 7'),
                   ('if stages and min(m,k,n) >= 16:', 'if stages:')],
    'verify_paths.py': [('assert min(m,k,n)>=16 and min(a,b,c)>=8', 'assert min(a,b,c)>=8'),
                        ("assert path.count(' / ')<=5", "assert path.count(' / ')<=7")]
}
inverses = []
for name, edits in changes.items():
    old = (OLD / name).read_text(); new = old
    for before, after in edits:
        assert new.count(before) == 1, (name, before)
        new = new.replace(before, after, 1)
    recovered = new
    for before, after in reversed(edits):
        assert recovered.count(after) == 1
        recovered = recovered.replace(after, before, 1)
    assert recovered == old
    ast.parse(new, filename=name)
    if (HERE / name).exists():
        assert (HERE / name).read_text() == new
    else:
        with (HERE / name).open('x') as f:
            f.write(new)
    inverses.append(dict(old=pin(OLD / name), new=pin(HERE / name), edits=edits,
                         whole_source_inverse_exact=True))
assert (HERE / 'reprice_inputs.json').read_bytes() == (OLD / 'reprice_inputs.json').read_bytes()
inputs = read(HERE / 'reprice_inputs.json')
external = [OLD / 'closeout.json', OLD / 'reprice_inputs.json', OLD / 'reprice_inputs_sealed.json',
    OLD / 'reprice_result.json', OLD / 'paths_verified.json', OLD / 'PROTOCOL.md', OLD / 'RESULT.md',
    OLD / 'reprice.py', OLD / 'verify_paths.py', D / 'continuation_checkpoint_v76.json',
    D / 'official_best_331684.json', D / 'run_guarded.py', PARENT,
    D / 'read6_packet_v1/NEGATIVE_RESULT.md', A / 'SMIRNOV_STREAM8_PROTOTYPE_V1_CLOSED.md',
    A / 'SMIRNOV_GLOBAL_BATCH_REPRESENTATIVE_V1_CLOSED.md']
external += [Path(r['path']) for r in inputs['censuses']]
external += [Path(inputs[k]['path']) for k in ['smirnov_certificate', 'catalog_certificate']]
for p in read(OLD / 'reprice_inputs_sealed.json')['pins']:
    check({k: p[k] for k in ['path', 'sha256', 'bytes']})
write(HERE / 'build.json', dict(source_inverses=inverses,
    old_study=pin(OLD / 'closeout.json'), previous_checkpoint=pin(D / 'continuation_checkpoint_v76.json'),
    arrays_not_decoded=True, modified_limits=dict(maximum_stages=[5, 7], parent_minimum=[16, None], child_minimum=8)))
for p in HERE.glob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
pins = [pin(p) for p in sorted(HERE.iterdir()) if p.is_file()] + [pin(p) for p in dict.fromkeys(external)]
write(HERE / 'reprice_inputs_sealed.json', dict(pins=pins, targets_not_decoded=True,
    before_new_calculation=True, no_new_model_calls=True))
write(HERE / 'sources_sealed.json', dict(pins=pins + [pin(HERE / 'reprice_inputs_sealed.json')]))
print(dict(prepared=True, source_files=len(pins), whole_source_inverses=2), flush=True)
