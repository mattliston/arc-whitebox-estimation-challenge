"""Freeze exact-arm sources and old metadata without opening label payloads."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
H2=HERE.parent
OLD=H2/'native_covariance_innovation_fit_v1'

def pin(path):
    path=Path(path).resolve();data=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))

def once(name,value):
    with (HERE/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')

own=[HERE/x for x in ('PROTOCOL.md','prepare_sources.py','seal_sources.py','common.py',
    'pair_base.py','projection.py','fit.py','fragment.py','build.py','unit.py','gradient.py',
    'evaluate.py','assess.py','launch_fit.py','launch_native.py','check_algebra.py',
    'algebra_verification.json','sketch_preparation_pins.json')]
for p in own:
    if p.suffix=='.py':ast.parse(p.read_text())
old=[OLD/x for x in ('PROTOCOL.md','RESULT.md','common.py','pair_base.py','projection.py','fit.py',
    'fragment.py','build.py','unit.py','gradient.py','labels/complete.json','assemble_guard.json',
    'references.json','fit_guard.json')]
old += [H2/'recurrent_covariance_observer_v1/capture/complete.json',
    H2/'current_covariance_pair_supervision_v1/capture/complete.json',
    H2/'terminal_contraction_actions_v1/PROTOCOL.md',H2/'terminal_contraction_actions_v1/RESULT.md',
    H2/'current_covariance_pair_supervision_v1/PROTOCOL.md',
    H2/'recurrent_covariance_observer_v1/PROTOCOL.md',
    H2/'candidate_old_both_tail_trace_bounded_v6.py',H2/'candidate_thin_star_route_v1.py',
    H2/'team_score_screen_v1.py',H2/'native_eval.py',H2/'compare_native.py',
    H2/'thin_star_route_train2_v1/manifest.json']
pins=[pin(p) for p in own+old]
once('seal.json',dict(sources=pins,source_only=True,label_payloads_opened=False,
    estimator_imported=False,numerical_jobs=0,feature='exact FP32 standardized-postCov shared-neighbor Gram RRt',
    arms=1,parameters=6209,epochs=40,updates=1680,sketch_executed=False))
once('fit_seal.json',dict(sources=pins))
once('native_seal.json',dict(sources=pins))
once('prior_sources.json',dict(sources=[pin(p) for p in old],
    nonlinear_global_pair_gram_executed_found=False,
    scope='Primary checkpoints42/43/44/47 and earlier index/team-log audit; not every archived source.'))
print(json.dumps(dict(seal=pin(HERE/'seal.json'),fit=pin(HERE/'fit.py'),
    protocol=pin(HERE/'PROTOCOL.md'),source_files=len(own),numerical_jobs=0)))
