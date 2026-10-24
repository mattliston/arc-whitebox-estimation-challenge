"""Source-only applicability review of the existing ordinary-client checks."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
D = HERE.parent
OLD = D/'frozen_covariance_readout_composition_v1'
FIT = D/'current_full_lower_p_fit_v1'
DOWN = D/'current_p_downstream_v1'


def pin(path):
    path = Path(path).resolve()
    b = path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))


def read(path):
    return json.loads(Path(path).read_text())


def check(item):
    assert pin(item['path']) == {k:item[k] for k in ['path','sha256','bytes']}


assert pin(DOWN/'sources_sealed.json')['sha256'] == '59f84db895a328e3552970f605aa860dff7857f5fbaf5043a116de64f4917e22'
sources = ['RELEASE_PROTOCOL.md','boundary_qualify.py','prepare_release.py','plain_driver.py','plain_client.py']
release_pins = read(OLD/'release_sources_sealed.json')['pins']
pins = [pin(OLD/name) for name in sources]
for p in pins:
    assert p in release_pins
assert read(OLD/'boundary_qualified.json')['passed']
copies = []
for name in ['plain_driver.py','plain_client.py']:
    prior = D/'readout_residual_runtime_v1'/name
    assert prior.read_bytes() == (OLD/name).read_bytes()
    copies.append(dict(prior=pin(prior),current=pin(OLD/name),byte_exact=True))
findings = []
for name in ['boundary_qualify.py','prepare_release.py']:
    source = (OLD/name).read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Assert):
            expression = ast.unparse(node.test)
            if ("x[:15]" in expression or "parent['flops'] == candidate['flops']" in expression):
                findings.append(dict(source=pin(OLD/name),line=node.lineno,assertion=expression,
                    valid_for_readout_only=True,required_for_current_all_p_fit=False))
assert len(findings) == 2 and any('x[:15]' in f['assertion'] for f in findings)
pins += [pin(OLD/'release_sources_sealed.json'),pin(OLD/'boundary_qualified.json'),
    pin(FIT/'ENDPOINT.md'),pin(FIT/'native.py'),pin(FIT/'score_native40.py'),
    pin(DOWN/'native.py'),pin(DOWN/'PROTOCOL.md'),pin(DOWN/'sources_sealed.json'),pin(HERE/'audit.py')]
assert not {'numpy','torch','flopscope'} & set(sys.modules)
record = dict(completed=True,utc=datetime.now(timezone.utc).isoformat(),source_only=True,
    pins=pins,unchanged_client_driver_copies=copies,inapplicable_readout_only_assertions=findings,
    current_fit_and_all_sealed_sources_changed=False,targets_opened=False,estimator_calls=0,
    client_or_backend_launched=False,release_protocol_declared=False,
    correct_invariants=['candidate own all16 native output bytes','candidate own complete native bills',
        'bank own all16 native output bytes and own bills','paired layer0 identity','same-history own repeats'],
    conditional_reference_gap=dict(if_reusing_old_13_client_call_design=True,
        cold_pairs=[35,100,37,39,42],warm_candidate=[35,100,35],
        new_native_reference_calls=4,reference_sequences=[[35,100,35],[100]],
        rationale='Candidate cold100 and short-history warm35 repeat are not acquired by the sealed native40/downstream rosters. Do not infer a cold bill from a warm bill or assume unrelated call histories have equal bills.',
        prerequisite='All fixed scientific gates must pass before declaring and executing a separate release study.'),
    old_local_runtime_margins=dict(pooled_wall_ratio_at_most=1.075,worst_wall_ratio_at_most=1.10,
        maximum_residual_seconds=.25,not_an_official_rule_or_new_all_p_release_protocol=True))
with (HERE/'audit.json').open('x') as f:
    json.dump(record,f,indent=2,allow_nan=False)
    f.write('\n')
print(dict(completed=True,source_only=True,inapplicable_assertions=len(findings),new_estimator_calls=0,
    conditional_native_reference_gap=4),flush=True)
