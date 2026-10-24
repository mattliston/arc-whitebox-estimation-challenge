"""Conditional fixed40 saved-score adapter; no execution before CHECK4 passes."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

_S40_HERE = Path(__file__).resolve().parent
_S40_OWNER = _S40_HERE/'score_check4_v2.py'
_S40_OWNER_SHA = '50279253cdd42322fdf0db7e73a29aa91b7807e8fa9e53b97c77e79c3192f076'
if hashlib.sha256(_S40_OWNER.read_bytes()).hexdigest() != _S40_OWNER_SHA:
    raise RuntimeError('Fixed CHECK4 v2 scoring source owner')
_S40_PARENT = runpy.run_path(str(_S40_OWNER),run_name='inert_score40_parent')
for _key,_value in _S40_PARENT.items():
    if not _key.startswith('__') and not _key.startswith('_S40'):
        globals()[_key] = _value
_S40_RANKS = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,139,145,
              146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234)


def _score40_prerequisite(admit,args):
    score,ss = admit.chain(HERE/'score_check4_v2',args.check4_score_sha256,120,1_000_000_000)
    watch,ws = admit.chain(HERE/'score_check4_watch_v2',args.check4_watch_sha256,120,1_000_000_000,watch=True)
    linked(watch,ss); admit.inventory(score); admit.inventory(watch)
    require(score['schema'] == 'current_comp_shared_old384_score_check4_v2'
        and score['ranks'] == [37,39,42,48] and len(score['records']) == 4
        and score['information_promising'] is True and score['scientific_gate_passed'] is True
        and watch['information_promising'] is True and score['comparison']['adjusted_ratio'] <= .97
        and score['comparison']['adjusted_wins'] >= 3 and score['comparison']['maximum_adjusted_ratio'] <= 1.10
        and score['complete_probe_acquisition_reverified'] and score['exact_APUSH_score_replay']
        and score['all_predictions_verified_before_truth'] and score['own_complete_bills']
        and score['full_model_calls'] == score['parameter_updates'] == 0 and score['repeat_excluded'],
        'Actual positive fixed CHECK4 objective gate, not a projection or negative acquisition')
    return score,dict(score=ss,watch=ws)


_S40_ORIGINAL = _V2_DERIVED
_old_line = next(s for s in _S40_ORIGINAL.splitlines() if s.startswith('TRUTHS = '))
_old_score_path = _S40_HERE.parent/'current_comp_aligned_push_d6_v1/score40_v1/result.json'
if hashlib.sha256(_old_score_path.read_bytes()).hexdigest() != 'b9667c17e474f99e394481169f9081d5e3320ca74255e2e3e411b7eaf9995279':
    raise RuntimeError('Original40 target-binding metadata')
_old_score = json.loads(_old_score_path.read_text())
if tuple(r['rank'] for r in _old_score['records']) != _S40_RANKS:
    raise RuntimeError('Original unchanged40 roster')
_targets = {str(_S40_HERE.parent.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{r:04d}.npz'):
    _old_score['records'][i]['target_sha256'] for i,r in enumerate(_S40_RANKS)}
_S40_EDITS = [
    ('RANKS = (37, 39, 42, 48)', 'RANKS = '+repr(_S40_RANKS), 1),
    (_old_line, 'TRUTHS = {Path(p):h for p,h in '+repr(_targets)+'.items()}', 1),
    ("'probe-protocol','probe-watch-spec','probe','probe-watch')", "'probe-protocol','probe-watch-spec','probe','probe-watch','check4-score','check4-watch')", 1),
    ("HERE/'check4_v2'", "HERE/'native40_v1'", 3),
    ("HERE/'check4_watch_v2'", "HERE/'native40_watch_v1'", 1),
    ('args.probe_sha256,660,8_000_000_000', 'args.probe_sha256,3000,8_000_000_000', 1),
    ('args.probe_watch_sha256,660,8_000_000_000', 'args.probe_watch_sha256,3000,8_000_000_000', 1),
    ("'current_comp_shared_old384_native_check4_v2'", "'current_comp_shared_old384_native40_v1'", 1),
    ("'current_comp_shared_old384_native_check4_watch_v2'", "'current_comp_shared_old384_native40_watch_v1'", 1),
    ("r['ranks'] == w['ranks'] == [37,39,42,48,37]", "r['ranks'] == w['ranks'] == list(RANKS)+[37]", 1),
    ("r['full_model_calls'] == r['retired_call_count'] == w['retired_calls'] == 5", "r['full_model_calls'] == r['retired_call_count'] == w['retired_calls'] == 41", 1),
    ("HERE/'check4_v2.py'", "HERE/'native40.py'", 2),
    ("HERE/'watch_check4_v2.py'", "HERE/'watch_native40.py'", 2),
    ("HERE/'CHECK4_V2_PROTOCOL.md'", "HERE/'NATIVE40_PROTOCOL.md'", 2),
    ("HERE/'CHECK4_V2_WATCH_SPEC.md'", "HERE/'NATIVE40_WATCH_SPEC.md'", 2),
    ("len(r['artifacts']) == 35 and len(r['trials']) == len(r['input_preparation']) == len(r['diagnostic_records']) == 5", "len(r['artifacts']) == 287 and len(r['trials']) == len(r['input_preparation']) == len(r['diagnostic_records']) == 41", 1),
    ("'All35 actual artifacts and five complete acquisition records'", "'All287 actual artifacts and41 complete acquisition records'", 1),
    ("HERE/'score_check4_v2.py'", "HERE/'score40.py'", 1),
    ("HERE/'watch_score_check4_v2.py'", "HERE/'watch_score40.py'", 1),
    ("HERE/'SCORE_CHECK4_V2_PROTOCOL.md'", "HERE/'SCORE40_PROTOCOL.md'", 1),
    ('PINS = {\n', 'PINS = {\n    _S40_OWNER:_S40_OWNER_SHA,\n', 1),
    ('prior,prerequisite = _check4_prerequisite(admit)', 'prior,prerequisite = _score40_prerequisite(admit,args)', 1),
    ("    result['score_metadata_revision_proof'] = _V2_PROOF\n", "    result['score_metadata_revision_proof'] = _V2_PROOF\n    result['score40_source_proof'] = _S40_PROOF\n", 1),
    ("[base['trials'][i] for i in (0,1,2,3,40)]", "list(base['trials'])", 1),
    ("for t in r['controls'][:4]", "for t in r['controls'][:40]", 1),
    ("for t in r['trials'][:4]", "for t in r['trials'][:40]", 1),
    ('for i in range(4))', 'for i in range(40))', 1),
    ('for x in v) < 1_000_000', 'for x in v) < 16_000_000', 1),
    ("'Eight distinct saved arrays below1MB'", "'Eighty distinct saved arrays below16MB'", 1),
    ('all_ten_outputs_verified', 'all_82_outputs_verified', 2),
    ("require(records[0] == prior['record'], 'Exact previously measured rank37 cold raw/cost/adjusted row')", "require(records[:4] == prior['records'], 'Exact previously measured first4 raw/cost/adjusted rows')", 1),
    ("sum(row['per_role'][role][key] for row in records)/4", "sum(row['per_role'][role][key] for row in records)/40", 1),
    ("promising = comparison['adjusted_ratio'] <= .97 and comparison['adjusted_wins'] >= 3 and max(ratios) <= 1.10", "promising = comparison['adjusted_ratio'] <= .97", 1),
    ("'All four original truths unchanged'", "'All40 original truths unchanged'", 1),
    ("'Mean of four per-net final raw*max(.1,own completeF/2^41); repeat excluded'", "'Mean of40 per-net final raw*max(.1,own completeF/2^41); repeat excluded'", 1),
    ("HERE/'score_check4_v2'", "HERE/'score40_v1'", 1),
    ("'current_comp_shared_old384_score_check4_v2'", "'current_comp_shared_old384_score40_v1'", 1),
]
# The complete-source inverse makes every metadata/scalar-roster change explicit.
_S40_DERIVED = _S40_ORIGINAL
for _before,_after,_count in _S40_EDITS:
    if _S40_DERIVED.count(_before) != _count:
        raise RuntimeError('Unique40 score span '+repr((_before,_S40_DERIVED.count(_before),_count)))
    _S40_DERIVED = _S40_DERIVED.replace(_before,_after)
_S40_INVERSE = _S40_DERIVED
for _before,_after,_count in reversed(_S40_EDITS):
    if _S40_INVERSE.count(_after) != _count:
        raise RuntimeError('Unique40 score inverse '+_after)
    _S40_INVERSE = _S40_INVERSE.replace(_after,_before)
if _S40_INVERSE != _S40_ORIGINAL:
    raise RuntimeError('Whole original CHECK4 v2 scorer restored')
ast.parse(_S40_DERIVED)
_S40_PROOF = dict(owner_sha256=_S40_OWNER_SHA,whole_source_inverse=True,
    derived_sha256=hashlib.sha256(_S40_DERIVED.encode()).hexdigest(),ranks=list(_S40_RANKS),
    unchanged_per_network_FP64_arithmetic=True,mean_of_products=True,repeat_excluded=True,
    first_four_scores_replay=True,prospective_mean_adjusted_ratio=.97)
exec(compile(_S40_DERIVED,str(_S40_OWNER)+':fixed40-metadata','exec'),globals())
