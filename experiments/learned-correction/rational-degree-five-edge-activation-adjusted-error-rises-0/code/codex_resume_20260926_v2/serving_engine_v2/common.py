"""Shared source-bound serving prerequisites; no job execution here."""
from pathlib import Path
import hashlib
import importlib.util
import json
import types

D = Path(__file__).resolve().parent
H2 = D.parent
P2 = H2.parent
ROOT = P2.parents[3]
LIGHT = ROOT/'.venv/bin/python'
NATIVE = P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
WHEST = ROOT/'.venv/bin/whest'
PREFIX = 'serving_v2_'
WORKER_RANKS = [35,100,37,39,42,48,59,63,66,67,71,73,76,80,84,88,92,96,104,108]
BROAD = H2/'valid40_engine_v2/assess.py'
BROAD_SHA = '3f833cc04ab41f5516562590a546b5113386e77d2cda4fc42f7f4566777d4f21'
GATES = dict(rss_bytes_below=7500000000,added_swap_bytes=0,handles_below=70000,
             index_caches=[320,352,320],setup_s_max=4.1,
             source_bound_residual_s_max=.35,projected_worst_wall_s_max=105)


def guard(ctx, path, seconds, command):
    assert seconds in (120,240,420,1050)
    g = ctx.q.read(path)
    assert g['command'] == command
    assert g['completed'] is True and g['exit_code'] == 0 and g['error'] is None
    assert g['whole_group_retired'] is True and g['remaining_rows'] == []
    assert g['rss_trigger_bytes'] == 7500000000
    assert 0 < g['max_rss_bytes'] < 7500000000 and g['max_swap_growth_bytes'] == 0
    assert 0 < g['elapsed_s'] <= seconds and g['samples'] > 0
    return g


def schedule(ctx):
    source=ctx.c['candidate']['path']; tag=ctx.c['release_tag']
    return [('worker',1050,[str(LIGHT),str(H2/'run_handle_worker_v3.py'),source,
                          ','.join(map(str,WORKER_RANKS)),tag+'_worker']),
            ('setup',120,[str(LIGHT),str(H2/'setup_stress_driver_v1.py'),source,tag+'_setup']),
            ('contract',120,[str(WHEST),'validate','--estimator',source,'--class','Estimator','--json'])]


def load_context(config_path,config_sha,admission_sha):
    assert hashlib.sha256(BROAD.read_bytes()).hexdigest() == BROAD_SHA
    spec=importlib.util.spec_from_file_location('serving_frozen_broad',BROAD)
    b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    q=b.initialize()
    cp=q.pin(config_path); assert cp['sha256']==config_sha
    c=json.loads(q.checked(cp)); study=Path(cp['path']).parent
    assert c['schema']=='serving-qualification-v2' and c['prospective_authorization'] is True
    assert c['common']==q.pin(__file__) and c['runner']==q.pin(D/'run.py')
    assert c['assessor']==q.pin(D/'assess.py') and c['broad_engine']==q.pin(BROAD)
    assert c['worker_ranks']==WORKER_RANKS and c['setup_trials']==3
    assert c['receipt_prefix']==PREFIX and c['release_tag']==study.name+'_serving_v2'
    assert c['resource_gates']==GATES and c['boundary_reused_without_rerun'] is True
    assert c['setup_counts'] and all(type(v) is int and v>0 for v in c['setup_counts'].values())
    assert c['guard_source']==q.pin(P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py')
    harness_paths=[str(WHEST)]+[str(H2/name) for name in (
        'run_handle_worker_v3.py','handle_server_v2.py','handle_client_v2.py',
        'setup_stress_driver_v1.py','setup_only_client_v1.py')]
    assert len(c['harnesses'])==len(harness_paths)
    assert {r['path'] for r in c['harnesses']}==set(harness_paths)
    for record in c['provenance']+c['harnesses']+[c['candidate'],c['policy'],c['common'],
            c['runner'],c['assessor'],c['broad_engine'],c['guard_source']]: q.checked(record)
    ap=q.pin(study/'validation_admission.json'); assert ap['sha256']==admission_sha
    a=json.loads(q.checked(ap))
    assert a['passed'] is True and a['candidate']==c['candidate'] and a['config']==cp
    assert a['policy']==c['policy'] and a['runner']==c['runner'] and a['assessor']==c['assessor']
    bp,bc=b.configure(q,Path(c['broad_config']['path']),c['broad_config']['sha256'])
    assert bp==c['broad_config'] and bc['candidate']['source']==c['candidate']
    assert bc['candidate']['setup']['counts']==c['setup_counts'] and bc['policy']==c['policy']
    assert a['broad_assessment']==q.pin(study/'broad_assessment.json')
    broad=json.loads(q.checked(a['broad_assessment']))
    assert broad['passed'] and broad['receipt_checks_passed'] and broad['validation_score_passed']
    assert broad['stage']=='VALID40' and broad['baseline_submission']==bc['score_reference']['submission_id']
    assert c['score_reference']==bc['score_reference']==broad['score_reference']
    assert c['runtime_reference']==bc['runtime_reference']==broad['runtime_reference']
    assert broad['baseline_adjusted']==bc['score_bank']['authenticated_poll']['score']
    assert broad['candidate']==c['candidate'] and broad['config']==bp and broad['assessor']==c['broad_engine']
    assert broad['policy']==c['policy'] and broad['gates'] and all(broad['gates'].values())
    metrics=broad['metrics']
    assert metrics['n']==40 and metrics['adjusted_pct']<=-1 and metrics['adjusted_wins']>=28
    assert metrics['adjusted_ci95_pct'][1]<0 and metrics['max_per_network_adjusted_regression_pct']<=10
    ctx=types.SimpleNamespace(q=q,c=c,config_pin=cp,admission_pin=ap,admission=a,
        study=study,H2=H2,P2=P2,ROOT=ROOT,LIGHT=LIGHT,NATIVE=NATIVE,WHEST=WHEST,
        PREFIX=PREFIX,WORKER_RANKS=WORKER_RANKS,broad=broad)
    assert a['broad_assess_guard']==q.pin(study/'broad_assess_guard.json')
    guard(ctx,study/'broad_assess_guard.json',120,
          [str(NATIVE),'-B',c['broad_engine']['path'],bp['path'],bp['sha256'],broad['admission']['sha256']])
    broad_ap,broad_a,runtime=b.prerequisites(q,bp,bc,broad['admission']['sha256'])
    assert broad['admission']==broad_ap
    assert broad['training']==broad_a['training'] and broad['boundary']==broad_a['boundary']
    assert broad['runtime_recheck']==runtime and runtime['passed'] is True
    for record in broad['evidence']: q.checked(record)
    ctx.training=json.loads(q.checked(broad['training']))
    ctx.boundary=json.loads(q.checked(broad['boundary']))
    ctx.runtime=runtime
    assert ctx.boundary['gates']['cpu_agreement'] is True
    assert ctx.boundary['residual']['projected_s']<=.35 and runtime['projected_worst_wall_s']<=105
    for label,count in c['setup_counts'].items():
        assert [r[1] for r in ctx.boundary['setup']['setup_log'] if r[0]==label]==[count]
    ctx.inputs=list(q.EVIDENCE.values())
    for record in ctx.inputs: q.checked(record)
    return ctx
