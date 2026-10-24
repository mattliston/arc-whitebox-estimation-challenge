"""Preparation path correction only; original failed script retained."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
s=(here/'prepare.py').read_text()
edits={"pending/'upload_auto_review_rejected.json'":"pending/'upload_review_rejection.json'","D/'full_tensor_exact_gram_information_v1/RESULT.md'":"D/'full_tensor_exact_gram_information_v1/NEGATIVE_RESULT.md'"}
for old,new in edits.items():assert s.count(old)==1;s=s.replace(old,new)
inverse=s
for old,new in reversed(list(edits.items())):assert inverse.count(new)==1;inverse=inverse.replace(new,old)
assert inverse==(here/'prepare.py').read_text()
with (here/'preparation_path_repair.json').open('x') as f:
    json.dump(dict(original_source_sha256=hashlib.sha256(inverse.encode()).hexdigest(),reason='Attempt1 raised FileNotFoundError while assembling source pins; no seal, numerical process or source_review.json was created.',edits=edits,inverse_exact=True,protocol_schedule_gates_unchanged=True),f,indent=2)
exec(compile(s,str(here/'prepare.py'),'exec'),dict(__file__=str(here/'prepare.py'),__name__='__main__'))
