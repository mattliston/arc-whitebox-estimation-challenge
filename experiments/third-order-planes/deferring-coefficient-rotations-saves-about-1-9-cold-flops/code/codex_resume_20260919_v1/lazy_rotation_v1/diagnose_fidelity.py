"""Inspect already sealed predictions after the fixed pre-target fidelity failure."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'weight5_reuse_v1'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = read(HERE/'build.json')
assert sha(HERE/'candidate.py') == build['candidate_sha256']
assert sha(PARENT/'candidate.py') == build['parent_sha256']
failure = read(HERE/'score_dev_watch.json')
assert failure['exit_code'] == 1 and failure['whole_group_retired']
assert not (HERE/'dev_predictions_sealed.json').exists()
assert not (HERE/'dev_score.json').exists()
records = []
for sequence in ('dev','cold100'):
    new = read(HERE/sequence/'result.json')
    old = read(PARENT/sequence/'result.json')
    assert new['candidate_sha256'] == build['candidate_sha256']
    assert old['candidate_sha256'] == build['parent_sha256']
    assert not new['truth_accessed']
    watch = read(HERE/(sequence+'_watch.json'))
    assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes'] == 0
    for a,b in zip(new['records'],old['records']):
        assert a['rank'] == b['rank'] and a['weights_sha256'] == b['weights_sha256']
        assert a['includes_initialization'] == b['includes_initialization']
        for item in (a,b):
            assert sha(item['output']) == item['output_sha256']
        x,y = np.load(a['output']),np.load(b['output'])
        assert x.shape == y.shape == (16,1024)
        assert np.isfinite(x).all() and (x >= 0).all()
        delta = x.astype(np.float64)-y.astype(np.float64)
        layer_maxabs = np.max(np.abs(delta),axis=1)
        layer_rms = np.sqrt(np.mean(delta**2,axis=1))
        records.append(dict(sequence=sequence,index=a['index'],rank=a['rank'],
                            includes_initialization=a['includes_initialization'],
                            output_sha256=a['output_sha256'],parent_output_sha256=b['output_sha256'],
                            weights_sha256=a['weights_sha256'],
                            first_eight_layers_byte_exact=np.array_equal(x[:8],y[:8]),
                            layerwise_maxabs=layer_maxabs.tolist(),layerwise_rms=layer_rms.tolist(),
                            maxabs=float(layer_maxabs.max()),
                            exceeding_1e_5_count=int(np.count_nonzero(np.abs(delta)>1e-5)),
                            fidelity_passed=bool(layer_maxabs.max()<=1e-5),
                            flops=a['flops'],parent_flops=b['flops'],
                            saved_flops=b['flops']-a['flops'],flop_ratio=a['flops']/b['flops'],
                            wall_s=a['wall_s'],native_peak_rss_bytes=a['max_rss_bytes']))
dev = read(HERE/'dev/result.json')['records']
cold = read(HERE/'cold100/result.json')['records'][0]
repeat = dev[0]['output_sha256'] == dev[2]['output_sha256']
warm = dev[1]['output_sha256'] == cold['output_sha256']
assert repeat and warm
assert all(r['first_eight_layers_byte_exact'] for r in records)
assert not all(r['fidelity_passed'] for r in records)
result = dict(candidate_sha256=build['candidate_sha256'],diagnostic_sha256=sha(__file__),
              records=records,repeat_byte_exact=repeat,warm_cold100_byte_exact=warm,
              first_eight_layers_byte_exact=True,fidelity_gate_passed=False,
              development_gate_passed=False,reference_means_opened=False,
              raw_or_adjusted_scores_measured=False,
              full_prediction_calls=4,billed_full_prediction_flops=sum(r['flops'] for r in records),
              interpretation='Complete native execution and cost reduction pass; fixed all-layer maxabs <=1e-5 fails. No claim of raw-score worsening or improvement.')
with (HERE/'fidelity_failure.json').open('x') as stream:
    json.dump(result,stream,indent=2)
print(json.dumps(result),flush=True)
