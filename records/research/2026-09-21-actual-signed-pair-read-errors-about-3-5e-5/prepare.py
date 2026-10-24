import ast
from common import *
assert sha(SOURCE)=='facd0dd0d3a425237a7285bd597e4f8ddb162f821f55acbba3bbaf136d80b6a0'
cp=D/'continuation_checkpoint_v89.json';assert sha(cp)=='b72f1e27f902c5e45659df2ec54b3c7cc1200f42108e41dca1b937467c64d565'
prior=read(cp)
for key in ('study','workers','verification','official_bank','pending_release','pending_candidate','pending_archive','rejection'):verify(prior[key])
old=D/'current_product_identity_census_v1/inputs.json';inputs=read(old)
assert inputs['source']==pin(SOURCE) and [r['rank'] for r in inputs['rows']]==[35,100]
external=[SOURCE,cp,old,D/'run_guarded.py',D/'official_best_331684.json']
for row in inputs['rows']:
    for key in ('weights','prediction','control','old_result','attribution'):
        verify(row[key]);external.append(Path(row[key]['path']))
external += [Path(prior[key]['path']) for key in ('study','workers','verification','pending_release','pending_candidate','pending_archive','rejection')]
external += [D/'polarized_pair_read_native_v1/RESULT.md',D/'polarized_pair_read_native_v1/wide_diagnostic.json',
    D/'shared_sketch_roundoff_diagnostic_v1/RESULT.md',D/'fp64_sketch_endpoint_v1/RESULT.md',
    P2/'astra_phase2_reexamination_v1/JOINT_CORE_AND_BINI_ERROR_SOURCE_AUDIT_2026_09_08_V1.md',
    P2/'astra_phase2_reexamination_v1/JOINT_CORE_AND_BINI_ERROR_SOURCE_AUDIT_2026_09_08_ADDENDUM_V1.md',
    P2/'astra_phase2_reexamination_v1/current_comp_receiver_birth_v1/BALANCE_RESULT.md']
for name in ('submission_intent.json','submission.json','submission_attempt.json'):
    assert not (D/'frozen_covariance_readout_composition_v1'/name).exists()
write(HERE/'inputs.json',dict(rows=inputs['rows'],source=pin(SOURCE),targets_not_read=True,pending_upload_unchanged=True))
for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
write(HERE/'sources_sealed.json',dict(pins=[pin(p) for p in sorted(HERE.iterdir()) if p.is_file()]+[pin(p) for p in dict.fromkeys(external)],
    before_units_and_new_calls=True,before_references=True,no_targets=True))
print(dict(sealed=True,ranks=[35,100],candidate_unchanged=True),flush=True)
