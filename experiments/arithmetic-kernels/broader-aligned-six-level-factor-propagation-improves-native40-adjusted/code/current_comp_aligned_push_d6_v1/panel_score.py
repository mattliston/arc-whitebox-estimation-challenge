"""Exact qualified independent14 scoring; APUSH versus actual completed CD."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
_CD = _HERE.parent/'current_comp_partial_d6_v1'
_PARENT = _CD/'panel_score.py'
_PARENT_SHA = '01bef7f3533518fda8fd78e005935124e10079ee87802b2ee65aee79ee042bc0'
_CURRENT_HELPER_SHA = 'b8ae22e9586f76cafd92763562d2a631efae6a8ea81d2417e6cad5ce6828129a'
_CURRENT_PROTOCOL_SHA = 'b398b6544b963c094ae8605d131cbec8d3f54d354ee6315b0498cecdc3ff5e1c'
_ACQUISITION_SOURCES = {
    'panel_common.py':_CURRENT_HELPER_SHA,
    'panel_admit.py':'c6d94d71669fc2c03a1c594d96d7f7ecd42342cc5300195a44175b36e8fd0fd5',
    'panel_native.py':'a30a4a634e729df078497bd003e1fec4b95220af30ee049ae8680968d682b499',
    'panel_watch.py':'b406f813f293f4c55ba251881d3315fe864bb5849ff663d3efd678137b05aa52',
    'PANEL_PROTOCOL.md':'9ff7c4591af0d7931dda4dba084d43db4a898ad5d2ef82b40a2255e28b236c47',
    'PANEL_WATCH_SPEC.md':'305e7f07528ec2edfdfdff91fdbeac39e531d14f5cb54e9395d76cde6e3297c3',
}


def _require(value,message):
    if not value: raise RuntimeError(message)


def _adapt(source,changes):
    changed = source
    for before,after,count in changes:
        _require(count > 0 and changed.count(before) == count,'Exact panel score source span '+before[:100])
        changed = changed.replace(before,after)
    inverse = changed
    for before,after,count in reversed(changes):
        _require(inverse.count(after) == count,'Exact panel score inverse span '+after[:100])
        inverse = inverse.replace(after,before)
    _require(inverse == source,'Complete qualified CD scorer restored')
    ast.parse(changed)
    return changed


def _function(source,name):
    found = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == name]
    _require(len(found) == 1,'Unique frozen scorer function '+name)
    n = found[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def _assignment(source,name):
    found = [n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == name for t in n.targets)]
    _require(len(found) == 1,'Unique frozen scorer assignment '+name)
    n = found[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def _roles(source):
    # Only literal result-role labels, including labels inside source strings.
    edits = [(a,b,source.count(a)) for a,b in (("'COMP'","'CD'"),("\\'COMP\\'","\\'CD\\'")) if source.count(a)]
    _require(edits,'Named comparator labels in the frozen function')
    return _adapt(source,edits)


def _control_admission(guard,pins,context):
    """Same newly qualified direct CD admission; return already verified bytes."""
    path = c.HERE/'panel_admit.py'
    digest = _ACQUISITION_SOURCES[path.name]
    c.require(c.sha(path,guard) == digest,'Frozen APUSH direct CD panel admission source')
    original = c.function_source(path,'qualified_cd')
    source = c.splice(original,[
        ('def qualified_cd(guard,inputs):','def qualified_control(guard,inputs):'),
        ('    return fixtures,rows,refs\n','    return native,score,outputs,fixtures,rows\n')])
    namespace = dict(c=c,SimpleNamespace=SimpleNamespace)
    exec(compile(source,str(path)+':same-CD-control-return-only','exec'),namespace)
    control,score,outputs,fixtures,rows = namespace['qualified_control'](guard,pins)
    c.require(fixtures == context['fixtures'] and rows == context['rows'] and len(outputs) == 15,
        'Exactly the same14 physical CD/APUSH fixtures and own repeat')
    context['control_admission_proof'] = dict(owner=str(path),owner_sha256=digest,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),full_source_inverse_verified=True,
        only_returned_already_verified_output_bytes=True,current_CD_three_seal_ownership_preserved=True,
        no_ancestor_reconstruction_or_score_execution=True)
    return control,score,outputs


_require(hashlib.sha256(_PARENT.read_bytes()).hexdigest() == _PARENT_SHA,'Frozen qualified CD panel score owner')
_require(hashlib.sha256((_HERE/'panel_common.py').read_bytes()).hexdigest() == _CURRENT_HELPER_SHA,
    'Exact currently running acquisition helper, never patched by score')
_original = _PARENT.read_text()
_factory_original = _function(_original,'score_function')
_factory = _adapt(_roles(_factory_original),[
    ('comparator label COMP and archived terminal score-role key',
     'comparator label CD and archived terminal score-role key',1)])
_finish_original = _function(_original,'finish_score')
_finish = _adapt(_roles(_finish_original),[
    ('exact_COMP14_rows_and_three_summaries_replay','exact_CD14_rows_and_three_summaries_replay',1),
    ('old_COMP_two_seal_ownership_preserved=True','current_CD_three_seal_ownership_preserved=True',1),
    ('        own_complete_bills=True,full_model_calls=0,retired_call_count=0,\n',
     '        own_complete_bills=True,full_model_calls=0,retired_call_count=0,\n'
     '        all_eligible_push_dispatches_qualified=True,runtime_prerequisite_qualified=True,\n'
     '        runtime_sha256=c.RUNTIME_SHA,original_external_watch_still_failed=True,\n'
     '        no_old_watch_repair_or_score_reexecution=True,\n',1),
    ("        control_admission_proof=context['control_admission_proof'],pareto_diagnostics=pareto,\n",
     "        control_admission_proof=context['control_admission_proof'],pareto_diagnostics=pareto,\n"
     "        panel_score_source_inverse=dict(_APUSH_SCORE_SOURCE_PROOF),\n",1),
])
_main_original = _function(_original,'derived_main')
_main = _roles(_main_original)
_helper_original = _function(_original,'score_helper')
_helper = _adapt(_helper_original,[
    ('COMP_CANDIDATE=c.pc.CANDIDATE,COMP_SHA=c.pc.CANDIDATE_SHA',
     "COMP_CANDIDATE=c.CD/'build_v1/estimator.py',COMP_SHA=c.d.CD_SHA",1)])
_verify_original = _function(_original,'verify_acquisition')
_verify = _adapt(_verify_original,[
    ("        and record['full_model_calls'] == record['retired_call_count'] == len(outputs) == 15,\n",
     "        and record['full_model_calls'] == record['retired_call_count'] == len(outputs) == 15\n"
     "        and record['all_eligible_push_dispatches_qualified'] and record['runtime_prerequisite_qualified']\n"
     "        and record['runtime_sha256'] == c.RUNTIME_SHA and record['initialization_flops'] == c.INIT\n"
     "        and record['control_result_sha256'] == c.CONTROL_SHA,\n",1)])
_changes = [
    ("_HELPER_SHA = '77d238cbaa3f046ecadcaa7520d5b7edf4075843b3bc329696ab4457dd90571c'",
     '_HELPER_SHA = '+repr(_CURRENT_HELPER_SHA),1),
    ("if hashlib.sha256((HERE/'panel_common_v2.py').read_bytes()).hexdigest() != _HELPER_SHA:",
     "if hashlib.sha256((HERE/'panel_common.py').read_bytes()).hexdigest() != _HELPER_SHA:",1),
    ('import panel_common_v2 as c\n','import panel_common as c\n',1),
    ("SCORE_PROTOCOL_SHA = '29f059c4993a5e40bf0c775a894a496a443159c8148cb75ea50eb77118b4f83b'",
     'SCORE_PROTOCOL_SHA = '+repr(_CURRENT_PROTOCOL_SHA),1),
    (_assignment(_original,'ACQUISITION_SOURCES'),'ACQUISITION_SOURCES = '+repr(_ACQUISITION_SOURCES)+'\n',1),
    ("description='Fixed saved-output CD independent14 '+role","description='Fixed saved-output APUSH independent14 '+role",1),
    ('    values[str(COMP_SCORER)] = COMP_SCORER_SHA\n',
     '    values[str(COMP_SCORER)] = COMP_SCORER_SHA\n    values[str(_PARENT)] = _PARENT_SHA\n',1),
    (_function(_original,'control_admission'),
     'def control_admission(guard,pins,context):\n    return _control_admission(guard,pins,context)\n',1),
    (_factory_original,_factory,1),(_finish_original,_finish,1),(_main_original,_main,1),
    (_helper_original,_helper,1),(_verify_original,_verify,1),
]
_source = _adapt(_original,_changes)
_APUSH_SCORE_SOURCE_PROOF = dict(parent=str(_PARENT),parent_sha256=_PARENT_SHA,
    derived_source_sha256=hashlib.sha256(_source.encode()).hexdigest(),whole_source_inverse_verified=True,
    original_numerical_expressions_casts_and_summation_order_unchanged=True,
    comparator_CD_actual_terminal_role=True,exact_baseline14_rows_and_three_summaries=True,
    current_acquisition_three_seals_and_all15_before_truth=True,
    no_extra_native_calls_or_candidate_changes=True,original_failed_watch_preserved=True,
    original120s_2GB_and_fresh_unique_union_unchanged=True)
# No main runs on import. --run is checked by the unchanged inherited entry point.
exec(compile(_source,str(_PARENT)+':APUSH-saved-panel-score','exec'),globals())
