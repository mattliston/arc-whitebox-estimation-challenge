"""Exclusive source assembly of three pinned components; stdlib only."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
S32 = H2/'hill_climb_star_projection32_v1/candidate.py'
UBC = H2/'hill_climb_uvp_base_bcache_v1'
D3 = H2/'hill_climb_star32_fused_d3_v1'
PB27 = H2/'hill_climb_s32_prebind27_v1'
EXPECTED = {
    S32: 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef',
    UBC/'candidate.py': '0308a99fd89823cfd67e19c3ad00c946d4211b6c69d956519267e142aa205fd5',
    D3/'candidate.py': '61a1f40b8a6e50d6c5e1b403a2200b0979d3d02794c0f522e29b98436c513c6e',
    PB27/'candidate.py': '1b7f024448cd5d05b1d92f8a9ba9d60c25c8d82750657a24dcb2872acc770457',
}


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def once(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def main():
    for path, digest in EXPECTED.items():
        assert pin(path)['sha256'] == digest
    parent = S32.read_bytes()
    base = (UBC/'candidate.py').read_bytes()
    d3 = (D3/'candidate.py').read_bytes()
    assert d3.startswith(parent)
    d3_delta = d3[len(parent):]
    pb_delta = (PB27/'fragment.py').read_bytes()
    assert (PB27/'candidate.py').read_bytes() == parent + pb_delta

    # Independently reverse the already pinned six UBC edits and append.
    ubc_build = json.loads((UBC/'build.json').read_text())
    ubc_fragment = (UBC/'fragment.py').read_bytes()
    assert base.endswith(b'\n\n' + ubc_fragment)
    inverse = base[:-len(ubc_fragment)-2].decode()
    for edit in reversed(ubc_build['edits']):
        assert inverse.count(edit['after']) == 1
        inverse = inverse.replace(edit['after'], edit['before'])
    assert inverse.encode() == parent

    old = None
    for node in ast.parse(parent).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_PB_SINGLE' for t in node.targets):
            try:
                old = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    def prefix_keys(fragment):
        node = ast.parse(fragment).body[0]
        assert isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp)
        return ast.literal_eval(node.value.left)
    nine, extra = prefix_keys(d3_delta), prefix_keys(pb_delta)
    assert len(nine) == 9 and len(extra) == 27 and len(old) == 600
    keys = extra + nine + old
    assert len(keys) == len(set(keys)) == 636
    counts = Counter(k[0] for k in keys)
    assert all(counts[k] < cap for k,cap in ((0,320),(1,352),(2,320)))
    candidate = base + d3_delta + pb_delta
    ast.parse(candidate)
    assert candidate[:-len(pb_delta)] == base + d3_delta
    assert candidate[:-len(pb_delta)-len(d3_delta)] == base
    assert candidate[len(base):len(base)+len(d3_delta)] == d3_delta
    assert candidate[-len(pb_delta):] == pb_delta
    d3_receipt = json.loads((D3/'probe_assessment_v2.json').read_text())
    ubc_receipt = json.loads((UBC/'retained_component.json').read_text())
    pb_receipt = json.loads((PB27/'retained_component.json').read_text())
    assert d3_receipt['passed'] and d3_receipt['candidate'] == pin(D3/'candidate.py')
    assert ubc_receipt['status'] == pb_receipt['status'] == 'RETAINED_EXACT_COMPONENT'
    assert ubc_receipt['candidate'] == pin(UBC/'candidate.py')
    assert pb_receipt['candidate'] == pin(PB27/'candidate.py')
    once(HERE/'d3_delta.py', d3_delta)
    once(HERE/'prebind27_delta.py', pb_delta)
    once(HERE/'candidate.py', candidate)
    report = dict(
        source_only=True, numerical_execution=False, status='UNQUALIFIED_EXACT_COMPONENT_COMPOSITION',
        candidate=pin(HERE/'candidate.py'), parent_s32=pin(S32), base=pin(UBC/'candidate.py'),
        order=['unchanged UVP B-cache candidate', 'exact D3 candidate suffix after S32', 'exact prebind27 fragment'],
        fragments=[pin(HERE/'d3_delta.py'), pin(HERE/'prebind27_delta.py')],
        component_sources=[pin(D3/'candidate.py'), pin(PB27/'candidate.py'), pin(PB27/'fragment.py'),
                           pin(UBC/'fragment.py'), pin(UBC/'build.json')],
        component_receipts=[pin(D3/'probe_assessment_v2.json'), pin(UBC/'retained_component.json'),
                            pin(PB27/'retained_component.json')],
        inverse_exact=dict(bundle_to_ubc=True, d3_delta_to_s32=True, prebind27_delta_to_s32=True,
                           ubc_six_edits_to_s32=True),
        no_duplicate_setup_keys=True, setup_keys=636, setup_bound_expected=660, setup_fused_expected=636,
        setup_expected_if='Fresh complete setup before unchanged four-second guard.',
        cache_counts_by_side={str(k): counts[k] for k in range(3)},
        cache_caps=[320,352,320], uvp_copy_limit_bytes=512000000,
        bundle_executed=False, bundle_qualified=False, broad_validation=False, submitted=False,
        component_gains_added=False, builder=pin(Path(__file__)), protocol=pin(HERE/'PROTOCOL.md'))
    once(HERE/'build.json', (json.dumps(report, indent=2)+'\n').encode())
    print(json.dumps({k:report[k] for k in ('candidate','setup_bound_expected','setup_fused_expected','cache_counts_by_side')}))


if __name__ == '__main__':
    main()
