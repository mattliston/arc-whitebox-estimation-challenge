from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
for p in read(HERE/'sources_sealed.json')['pins']:assert pin(p['path'])==p
g=read(HERE/'score_watch.json');r=read(HERE/'score.json')
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
live={int(x) for x in subprocess.check_output(['/bin/ps','-axo','pgid='],text=True).split()}
assert g['pid'] not in live
assert not r['passed']
text=f'''# Frozen FP64 sketch endpoint: negative adjusted objective

This separate fixed screen reused only the two unperturbed cold predictions
from fp64_sketch_response_v1. No new estimator calls or fitting occurred.
Source/prediction/weight/target pins, complete cold bills, cache retirement,
finite nonnegative outputs and first5 parent bytes pass qualification.

| Rank | Raw ratio | Cost ratio | Adjusted ratio |
| --- | --- | --- | --- |
'''
for x in r['rows']:
    text+=f"| {x['rank']} | {x['raw_ratio']:.12g} | {x['cost_ratio']:.12g} | {x['adjusted_ratio']:.12g} |\n"
text+=f'''
Pooled raw ratio{r['pooled_raw_ratio']:.12g}; pooled adjusted ratio
{r['pooled_adjusted_ratio']:.12g}. The frozen candidate is about11.0% WORSE
on the actual adjusted objective of these reused public research networks.
Raw error is essentially unchanged; the added10.9% bill is not repaid.
This is not an official grade or pristine validation. Both the accuracy gate
and original response gate fail independently. No CHECK/panel/confirmation,
fit, client/package or upload follows this candidate. No precision/parameter
tuning is selected by this result. Better numerical sensitivity alone has not
delivered better estimation accuracy.

See CHRONOLOGY_CORRECTION.md: declaration was after response result creation
but before observing it and before this screen's target decoding. The initial
"still running" description was based on the last live poll and is corrected.
Original sources and seals remain intact.

One guarded scalar-score process retired; zero added swap, no new samples,
downloads or GPU. Official331684 remains3.5844144212297072e-9; goal active.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(text)
files=sorted(p for p in HERE.iterdir() if p.is_file())
write(HERE/'closeout.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
 artifact_pins=[pin(p) for p in files],external_pins=read(HERE/'sources_sealed.json')['pins'],
 outcome=r,guard=g,all_workers_retired=True,new_full_calls=0,new_fit_steps=0,new_samples=0,goal_active=True))
print(dict(completed=True,passed=r['passed'],pooled_adjusted_ratio=r['pooled_adjusted_ratio'],pins=len(files)),flush=True)
