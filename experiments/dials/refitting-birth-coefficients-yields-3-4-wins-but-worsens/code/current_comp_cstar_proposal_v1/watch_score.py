"""Exact bounded build-watch transport reused for one saved-output score."""
import ast
import hashlib
from pathlib import Path

_HERE=Path(__file__).resolve().parent
_OWNER=_HERE/'watch_build.py'
_OWNER_SHA='f7db9500042abcf0157593ff93fc3e9cd47f3db340bdf7b4bd1dc90bf199ea1b'
_original=_OWNER.read_text()
if hashlib.sha256(_original.encode()).hexdigest()!=_OWNER_SHA:
    raise RuntimeError('Exact bounded source-build supervisor owner')

_changes=[
    ("SCHEMA = 'current_comp_cstar_proposal_build_watch_v1'",
     "SCHEMA = 'current_comp_cstar_proposal_score_watch_v1'"),
    ("    keys = ('source','builder','protocol','spec','qualification-result','qualification-watch',\n"
     "        'training-result','training-watch')\n",
     "    keys = ('source','scorer','protocol','spec','native-source','native-watch-source',\n"
     "        'native-protocol','native-watch-spec','builder','build-protocol','build-watch',\n"
     "        'build-watch-source','build-watch-spec','terminal-build','terminal-candidate',\n"
     "        'terminal-native','terminal-watch')\n"),
    ("    own = {Path(__file__).resolve():args.source_sha256,HERE/'build.py':args.builder_sha256,\n"
     "        HERE/'BUILD_PROTOCOL.md':args.protocol_sha256,HERE/'BUILD_WATCH_SPEC.md':args.spec_sha256,\n"
     "        TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}\n",
     "    own = {Path(__file__).resolve():args.source_sha256,HERE/'score.py':args.scorer_sha256,\n"
     "        HERE/'SCORE_PROTOCOL.md':args.protocol_sha256,HERE/'SCORE_WATCH_SPEC.md':args.spec_sha256,\n"
     "        TRANSPORT:TRANSPORT_SHA,_OWNER:_OWNER_SHA,**h.MON_PINS}\n"),
    ("    for folder,digest in (('qualification_v1',args.qualification_result_sha256),\n"
     "            ('qualification_watch_v1',args.qualification_watch_sha256),\n"
     "            ('training_v1',args.training_result_sha256),('training_watch_v1',args.training_watch_sha256)):\n",
     "    for folder,digest in (('build_v1',args.terminal_build_sha256),\n"
     "            ('build_watch_v1',args.build_watch_sha256),\n"
     "            ('native_terminal_v1',args.terminal_native_sha256),\n"
     "            ('native_terminal_watch_v1',args.terminal_watch_sha256)):\n"),
    ("    folder,out = HERE/'build_v1',HERE/'build_watch_v1'\n",
     "    folder,out = HERE/'score_v1',HERE/'score_watch_v1'\n"),
    ("        forwarded = dict(source=args.builder_sha256,protocol=args.protocol_sha256,\n"
     "            watcher=args.source_sha256,watch_spec=args.spec_sha256,\n"
     "            qualification_result=args.qualification_result_sha256,qualification_watch=args.qualification_watch_sha256,\n"
     "            training_result=args.training_result_sha256,training_watch=args.training_watch_sha256)\n"
     "        argv = [str(OFFPY),'-B','-u',str(HERE/'build.py'),'--build']\n",
     "        forwarded = {key.replace('-','_'):getattr(args,key.replace('-','_')+'_sha256')\n"
     "            for key in keys if key not in ('source','scorer','spec')}\n"
     "        forwarded.update(source=args.scorer_sha256,watcher=args.source_sha256,watch_spec=args.spec_sha256)\n"
     "        argv = [str(OFFPY),'-B','-u',str(HERE/'score.py'),'--run']\n"),
    ("        require(r['schema'] == 'current_comp_cstar_proposal_build_v1'\n"
     "            and r['hashes_unchanged'] and r['sources_unchanged'] and r['parameter_updates'] == 0\n"
     "            and r['estimator_executed'] is False and r['truth_arrays_accessed'] is False\n"
     "            and r['peak_rss_bytes'] < 2_000_000_000,'Sole bounded literal construction, no model or fitting')\n",
     "        require(r['schema'] == 'current_comp_cstar_proposal_score_v1'\n"
     "            and r['hashes_unchanged'] and r['sources_unchanged']\n"
     "            and all(r.get(k) is True for k in ('complete_cstar_acquisition_reverified',\n"
     "                'source_build_watch_qualified','all_five_domains_and_selectors_reverified',\n"
     "                'all_predictions_verified_before_truth','exact_COMP_score_replay',\n"
     "                'no_model_imported','no_native_gradient_claim'))\n"
     "            and type(r['cstar_component']) is bool and r['advances_to_native40'] == r['cstar_component']\n"
     "            and r['peak_rss_bytes'] < 2_000_000_000,\n"
     "            'Complete sole fixed score; scientific failure is not operational failure')\n"),
    ("        required_owners = {HERE/'build.py':args.builder_sha256,HERE/'BUILD_PROTOCOL.md':args.protocol_sha256,\n"
     "            Path(__file__).resolve():args.source_sha256,HERE/'BUILD_WATCH_SPEC.md':args.spec_sha256}\n",
     "        required_owners = {HERE/'score.py':args.scorer_sha256,HERE/'SCORE_PROTOCOL.md':args.protocol_sha256,\n"
     "            Path(__file__).resolve():args.source_sha256,HERE/'SCORE_WATCH_SPEC.md':args.spec_sha256}\n"),
    ("            candidate=r['candidate'],verified_child_nontarget_files=len(union),\n",
     "            candidate_sha256=args.terminal_candidate_sha256,verified_child_nontarget_files=len(union),\n"
     "            cstar_component=r['cstar_component'],advances_to_native40=r['advances_to_native40'],\n"
     "            scientific_advancement=r['cstar_component'],\n"
     "            score_supervisor_source_inverse=SOURCE_INVERSE_PROOF,\n"),
    ("        'maximum_interval_s','candidate','error')},allow_nan=False),flush=True)\n",
     "        'maximum_interval_s','candidate_sha256','cstar_component','error')},allow_nan=False),flush=True)\n"),
]
_derived=_original
for _before,_after in _changes:
    if _derived.count(_before)!=1:raise RuntimeError('Unique score-watch adaptation: '+_before[:100])
    _derived=_derived.replace(_before,_after)
_inverse=_derived
for _before,_after in reversed(_changes):
    if _inverse.count(_after)!=1:raise RuntimeError('Unique score-watch inverse: '+_after[:100])
    _inverse=_inverse.replace(_after,_before)
if _inverse!=_original:raise RuntimeError('Whole original resource supervisor restored')
ast.parse(_derived)
SOURCE_INVERSE_PROOF=dict(owner=str(_OWNER),owner_sha256=_OWNER_SHA,whole_source_inverse_verified=True,
    original_sha256=hashlib.sha256(_original.encode()).hexdigest(),
    derived_sha256=hashlib.sha256(_derived.encode()).hexdigest(),
    resource_transport_deadlines_and_cleanup_unchanged=True,metadata_and_fixed_score_command_only=True)
exec(compile(_derived,str(_OWNER)+':saved-output-score-metadata-only','exec'),globals())
