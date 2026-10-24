"""Fixed APUSH native40 watch; versioned roster and actual-schema inventory repair."""
import ast
import hashlib
from pathlib import Path

_here=Path(__file__).resolve().parent
_parent=_here/'watch_native.py'
_parent_sha='dca03cd8830f02faee4757383e541e759bc0170cb9299cc60d1283326af31064'
_ranks=(37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
    139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234,37)
if hashlib.sha256(_parent.read_bytes()).hexdigest()!=_parent_sha:
    raise RuntimeError('Frozen original native5 watch, including its known inventory failure, preserved')
_original=_parent.read_text()
_changes=(
    ('RANKS=(37,39,42,48,37)','RANKS='+repr(_ranks),1),
    ('CAP,RSS=600.,8_000_000_000','CAP,RSS=1900.,8_000_000_000',1),
    ("keys=('source','worker','protocol','spec','runtime','build','candidate')",
     "keys=('source','worker','protocol','spec','runtime','build','candidate','check4-score','check4-watch')",1),
    ("getattr(args,k+'_sha256')","getattr(args,k.replace('-','_')+'_sha256')",1),
    ("HERE/'native.py'","HERE/'native40.py'",4),
    ("HERE/'NATIVE_PROTOCOL.md'","HERE/'NATIVE40_PROTOCOL.md'",1),
    ("HERE/'NATIVE_WATCH_SPEC.md'","HERE/'NATIVE40_WATCH_SPEC.md'",1),
    ("HERE/'build_v1/estimator.py':CANDIDATE_SHA,TRANSPORT:TRANSPORT_SHA,**h.MON_PINS}",
     "HERE/'build_v1/estimator.py':CANDIDATE_SHA,TRANSPORT:TRANSPORT_SHA,**h.MON_PINS,\n"
     "        HERE/'watch_native.py':'"+_parent_sha+"'}",1),
    ("HERE/'build_v1/result.json':args.build_sha256}",
     "HERE/'build_v1/result.json':args.build_sha256,\n"
     "        HERE/'score_v2/result.json':args.check4_score_sha256,\n"
     "        HERE/'score_watch_v1/result.json':args.check4_watch_sha256}",1),
    ("out=HERE/'native_watch_v1';ad=HERE/'native_terminal_admission_v1';native=HERE/'native_terminal_v1'",
     "out=HERE/'native40_watch_v1';ad=HERE/'native40_terminal_admission_v1';native=HERE/'native40_terminal_v1'",1),
    ("schema='current_comp_aligned_push_d6_native_watch_v1'","schema='current_comp_aligned_push_d6_native40_watch_v1'",1),
    ("'Whole600s/next-retired-context120s'","'Whole1900s/next-retired-context120s'",1),
    ("index<5","index<41",1),
    ("'Exactly ordered fully retired native5'","'Exactly ordered fully retired native41'",1),
    ("            require(isinstance(record[group],dict) and record[group],'Complete non-target inventory')\n"
     "            for p,digest in record[group].items():",
     "            require(isinstance(record.get(group,{}),dict)\n"
     "                and (group=='input_bindings' or record.get(group)),'Actual non-target inventory schema')\n"
     "            for p,digest in record.get(group,{}).items():",1),
    ("'--candidate-sha256',CANDIDATE_SHA]",
     "'--candidate-sha256',CANDIDATE_SHA,\n"
     "            '--check4-score-sha256',args.check4_score_sha256,'--check4-watch-sha256',args.check4_watch_sha256]",1),
    ("result['retired_call_count']==5","result['retired_call_count']==41",1),
    ("'Native worker exited with all five fully retired'","'Native worker exited with all41 fully retired'",1),
    ("record['schema']=='current_comp_aligned_push_d6_native_v1'",
     "record['schema']=='current_comp_aligned_push_d6_native40_v1'",1),
    ("record['full_model_calls']==record['retired_call_count']==5",
     "record['full_model_calls']==record['retired_call_count']==41",1),
    ("'Actual complete fixed five-call native subject'","'Actual complete fixed41-call native subject'",1),
    ("                    'sources_unchanged','hashes_unchanged'):",
     "                    'sources_unchanged','hashes_unchanged','check4_expansion_prerequisite_passed',\n"
     "                    'check4_first4_output_byte_exact','check4_first4_deterministic_bills_exact',\n"
     "                    'check4_first4_domains_exact','check4_first4_dispatches_exact'):",1),
    ("whole_owned_tree_retired=True,full_model_calls=5,","whole_owned_tree_retired=True,full_model_calls=41,",1),
    ("all5_complete=True","all41_complete=True",1),
)
_derived=_original
for _before,_after,_count in _changes:
    if _derived.count(_before)!=_count: raise RuntimeError('Unique roster/schema native40 watch span: '+_before)
    _derived=_derived.replace(_before,_after)
_inverse=_derived
for _before,_after,_count in reversed(_changes):
    if _inverse.count(_after)!=_count: raise RuntimeError('Unique inverse native40 watch span: '+_after)
    _inverse=_inverse.replace(_after,_before)
if _inverse!=_original: raise RuntimeError('Entire original watch restored')
ast.parse(_derived)
exec(compile(_derived,str(_parent)+':fixed-native40-and-inventory-schema','exec'),
    dict(__name__=__name__,__file__=__file__))
