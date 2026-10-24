"""Genuine historical used651 baseline/zero-envelope identity; no gradients."""
import contextlib
import hashlib
import json
from pathlib import Path
import runpy
import sys

T = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(T))
import otr_edband_v1 as EB
import edband_six_arms_v3 as A

assert Path.cwd().resolve() == T
OUT = T/'cert/edband_used651_pair_v1'
OUT.mkdir(exist_ok=False)
PARAMS = T/'cert/refit/jmI_p2_noAS.pt'
HEAD = T/'cert/refit/jmI_p2_hu.json'
REQUEST = T/'cert/cutsim_s46_20261010T232518Z/frozen_v11_sources/a_cutsim13_v11right_r0651.request.json'
WORKER = T/'worker_edband_v2.py'
assert EB.sha256(PARAMS) == 'f36e034c6196f7693b4e01c453f7e2044aca9b69010bd2f4ab5c2340022411e3'
assert EB.sha256(HEAD) == 'd5b16582cfec6f84ae3f50d75e3d854c91585da5284a83f3742a45af7e40fa7c'
assert EB.sha256(WORKER) == A.WORKER_SHA
head = json.loads(HEAD.read_text())
base_request = json.loads(REQUEST.read_text())
assert base_request['rank'] == 651 and base_request['mode'] == 'eval'
assert Path(base_request['params']).resolve() == PARAMS and base_request['hu'] == head
assert base_request['cutsim']['opts'] == dict(enabled=True,law_response_fit=True,law_beta=True,law_restore_right=True)
assert base_request.get('edge_band') is None and base_request.get('truth_plan') is None
pins = {name:A.pin(path) for name,path in dict(params=PARAMS,head=HEAD,request=REQUEST,worker=WORKER,
                                            adapter=Path(EB.__file__),helper=Path(__file__)).items()}
import torch
torch.set_num_threads(3)
base = torch.load(PARAMS,map_location='cpu',weights_only=True)
payload = EB.make_checkpoint(torch,base,source=dict(purpose='historical_used651_zero_identity',sources=pins))
saved = EB.save_checkpoint(torch,OUT/'zero_envelope.pt',payload)
assert all(torch.equal(value,torch.zeros_like(value)) for value in payload['edge_band']['offsets'].values())
baseline = dict(base_request,id='edband651_0baseline',keep_all=1)
baseline.pop('keep_final',None)
zero = dict(baseline,id='edband651_1zero',params=saved['path'],
            edge_band=EB.request_spec(saved['path'],saved['sha256']))
for request in (baseline,zero):
    assert request['mode']=='eval' and request['rank']==651 and 'scale' not in request
for directory in ('claimed','queue','results'):(OUT/directory).mkdir()
(OUT/'STOP').touch(exist_ok=False)
for request in (baseline,zero):A.write_json(OUT/'claimed'/(request['id']+'.w0.json'),request)
A.write_json(OUT/'head.json',head)
A.write_json(OUT/'BASELINE_REQUEST.json',baseline)
A.write_json(OUT/'ZERO_REQUEST.json',zero)
A.write_json(OUT/'STARTED.json',dict(schema='trainer_edband_used651_identity_v1',sources=pins,threads=3,
                                  zero_envelope=saved,requested_network_evaluations=2,gradients=0))
sys.argv=[str(WORKER),str(OUT),'0']
with (OUT/'worker.log').open('x') as stream, contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
    runpy.run_path(str(WORKER),run_name='__main__')
records=[json.loads((OUT/'results'/(request['id']+'.json')).read_text()) for request in (baseline,zero)]
for request,record in zip((baseline,zero),records):
    for key in ('id','rank','mode','params','cutsim'):assert record[key]==request[key],key
    assert record['threads']==3 and record['data_source']=='public' and record.get('truth_metadata') is None
    assert record['edge_band_worker']==A.expected_worker()
    assert len(record['prediction'])==16 and all(len(row)==1024 for row in record['prediction'])
    assert 'edge_band_grads' not in record
assert records[0]['edge_band']==dict(schema='trainer_edge_band_provenance_v1',enabled=False,
                                   module_sha256=EB.sha256(EB.__file__),request=baseline)
assert records[1]['edge_band']==EB.Session(torch,zero['edge_band'],zero)._identity
arrays=[torch.tensor(record['prediction'],dtype=torch.float32) for record in records]
assert all(bool(torch.isfinite(value).all()) for value in arrays)
assert arrays[0].contiguous().view(torch.uint8).numpy().tobytes()==arrays[1].contiguous().view(torch.uint8).numpy().tobytes()
assert records[0]['mse']==records[1]['mse'] and records[0]['mse_layers']==records[1]['mse_layers']
for item in pins.values():assert EB.sha256(item['path'])==item['sha256']
receipt=dict(schema='trainer_edband_used651_identity_v1',status='PASS_USED651_IDENTITY_ONLY',sources=pins,
    baseline=A.pin(OUT/'results'/(baseline['id']+'.json')),zero=A.pin(OUT/'results'/(zero['id']+'.json')),
    baseline_request=A.pin(OUT/'BASELINE_REQUEST.json'),zero_request=A.pin(OUT/'ZERO_REQUEST.json'),
    zero_envelope=saved,head=A.pin(OUT/'head.json'),threads=3,rank=651,
    all16_predictions_bit_exact=True,all16_mse_exact=True,mse=records[0]['mse'],
    gradients=0,generated_truth_reads=0,latest_best_qualified=False,native_qualified=False)
A.write_json(OUT/'IDENTITY.json',receipt)
print(json.dumps(dict(status=receipt['status'],receipt=str(OUT/'IDENTITY.json'))))
