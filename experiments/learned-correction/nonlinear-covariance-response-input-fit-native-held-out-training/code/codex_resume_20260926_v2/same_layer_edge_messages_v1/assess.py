"""Fixed CHECK8 assessment; no fitted amplitude, epoch or arm selection."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
HERE=Path(__file__).resolve().parent;H2=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((HERE/'protocol.json').read_text());ranks=protocol['check_ranks']
matched=sys.argv[1:]==['--matched-baseline'];assert not sys.argv[1:] or matched
output=HERE/('assessment_matched.json' if matched else 'assessment.json')
assert not output.exists()
base={};refs={}
base_tags=('same_layer_edge_baseline_check8_v1',) if matched else ('both_tail_train2_v6','both_tail_train6_v6_routing_control')
if matched:
    g=json.loads((HERE/'matched_baseline_guard.json').read_text())
    assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
for tag in base_tags:
    path=H2/tag;m=json.loads((path/'manifest.json').read_text())
    assert m['sha256']==protocol['parent_sha256']==sha(Path(m['candidate']))
    for r in map(json.loads,(path/'rows.jsonl').read_text().splitlines()):
        assert r['rank'] not in base;base[r['rank']]=r;refs[r['rank']]=path
assert set(base)==set(ranks)
baseline_parity=[]
if matched:
    for tag in ('both_tail_train2_v6','both_tail_train6_v6_routing_control'):
        old=H2/tag
        for prior in map(json.loads,(old/'rows.jsonl').read_text().splitlines()):
            rank=prior['rank']
            assert np.array_equal(np.load(old/f'pred_{rank}.npy'),np.load(refs[rank]/f'pred_{rank}.npy')),rank
            assert prior['mse']==base[rank]['mse'],rank
            baseline_parity.append(dict(rank=rank,all16_bytes_exact=True,
                old_flops=prior['flops'],matched_flops=base[rank]['flops'],
                matched_bill_pct=100*(base[rank]['flops']/prior['flops']-1)))
arms={};sources={};first8={}
for mode in ('message','control'):
    path=H2/f'same_layer_edge_{mode}_check8_v1';m=json.loads((path/'manifest.json').read_text())
    assert m['sha256']==sha(Path(m['candidate']))
    assert m['sha256']==json.loads((HERE/f'build_{mode}_terminal.json').read_text())['sha256']
    rows={r['rank']:r for r in map(json.loads,(path/'rows.jsonl').read_text().splitlines())};assert set(rows)==set(ranks)
    sources[mode]=m;arms[mode]=rows;first8[mode]=[]
    guard=json.loads((HERE/f'check_{mode}_guard.json').read_text())
    assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
    for rank in ranks:
        y=np.load(path/f'pred_{rank}.npy');b=np.load(refs[rank]/f'pred_{rank}.npy')
        assert y.shape==b.shape==(16,1024) and np.isfinite(y).all()
        assert np.array_equal(y[:8],b[:8]),(mode,rank,'early layers changed')
        first8[mode].append(rank)
rng=np.random.default_rng(20260926);idx=rng.integers(0,len(ranks),size=(50000,len(ranks)))
def compare(left,right):
    keys=('mse','adjusted','flops','cpu_s')
    a=np.array([[left[r][k] for k in keys] for r in ranks]);b=np.array([[right[r][k] for k in keys] for r in ranks])
    assert np.isfinite(a).all() and np.isfinite(b).all()
    ratio=a[:,:2][idx].mean(1)/b[:,:2][idx].mean(1)-1
    return dict(raw_pct=100*(a[:,0].mean()/b[:,0].mean()-1),adjusted_pct=100*(a[:,1].mean()/b[:,1].mean()-1),
        raw_ci95_pct=(100*np.quantile(ratio[:,0],[.025,.975])).tolist(),
        adjusted_ci95_pct=(100*np.quantile(ratio[:,1],[.025,.975])).tolist(),
        adjusted_wins=int((a[:,1]<b[:,1]).sum()),raw_wins=int((a[:,0]<b[:,0]).sum()),
        bill_pct=100*(a[:,2].mean()/b[:,2].mean()-1),cpu_pct=100*(a[:,3].mean()/b[:,3].mean()-1),
        mean_adjusted=float(a[:,1].mean()),reference_mean_adjusted=float(b[:,1].mean()),
        per_network=[dict(rank=r,raw_pct=100*(left[r]['mse']/right[r]['mse']-1),
                         adjusted_pct=100*(left[r]['adjusted']/right[r]['adjusted']-1)) for r in ranks])
message=compare(arms['message'],base);control=compare(arms['control'],base);contrast=compare(arms['message'],arms['control'])
gates=dict(at_least_two_percent_vs_bank=message['adjusted_pct']<=-2,
    at_least_six_wins=message['adjusted_wins']>=6,bootstrap_upper_below_zero=message['adjusted_ci95_pct'][1]<0,
    at_least_one_percent_vs_local_control=contrast['adjusted_pct']<=-1)
gates={k:bool(v) for k,v in gates.items()}
result=dict(ranks=ranks,fit_ranks=protocol['fit_ranks'],message_vs_bank=message,control_vs_bank=control,
    message_vs_control=contrast,gates=gates,passes_pilot=all(gates.values()),first8_exact=first8,
    protocol_sha256=sha(HERE/'protocol.json'),assessment_sha256=sha(HERE/'assess.py'),sources=sources,
    valid40_accessed=False,board_submitted=False,baseline_tags=base_tags,matched_worker_order=matched,baseline_parity=baseline_parity,
    comparison_scope='Final matched eight-network worker order' if matched else 'Historical split baseline; fresh matched baseline assessment is required before a final gain claim',
    decision='Advance to a separately fixed larger qualification' if all(gates.values()) else 'Close this fixed pilot; no amplitude, epoch, sign or feature rescue')
serialized=json.dumps(result,indent=2,allow_nan=False)
with output.open('x') as fp:fp.write(serialized+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('sources','first8_exact')},indent=2))
