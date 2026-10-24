"""Locate drift in already sealed outputs; never rerun or overturn the gate.

Fixed scope: both recorded ranks and all five cases. Report layerwise primal
and symmetric-response vector differences only. No new scalar, threshold,
seed, amplitude, targets, weights, model calls, fitting or score judgment.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time

HERE = Path(__file__).resolve().parent
HY = HERE.parent/'current_d50_hybrid_cp_qualification_v1'
CASES = ('null','plus025','minus025','plus050','minus050')
JUDGE_SHA = 'a8ff76e5d020a59da6626d52b79993c9e28da118abdc458ff9bbcc1c9372f4f5'
HY_SHA = 'd6edae6b5828e09939d5afb7be7a3bc619f8e384fd909cbf105b26331a2943fc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.getpriority(os.PRIO_PROCESS,0)>=10
    assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
    start = time.monotonic()
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        assert path.is_relative_to(HERE) or path.is_relative_to(HY)
        digest = sha(path)
        assert expected is None or digest==expected, path
        pins[str(path)] = digest
        return path

    pin(Path(__file__))
    jp = pin(HERE/'judgment_v1/result.json',JUDGE_SHA)
    hp = pin(HY/'stage_mps_forward_v1/result.json',HY_SHA)
    judge,hybrid = json.loads(jp.read_text()),json.loads(hp.read_text())
    assert judge['passed'] is False and judge['null_bytes_and_bills_exact']
    assert hybrid['passed'] and hybrid['completed']
    native = {}
    for case in CASES:
        nr = judge['native_records'][case]
        folder = HERE/('run_'+case+'_v1')
        rp = pin(folder/'result.json',nr['result_sha256'])
        cp = pin(folder/'completion.json')
        record,seal = json.loads(rp.read_text()),json.loads(cp.read_text())
        assert record['passed'] and seal['passed'] and seal['result_sha256']==sha(rp)
        native[case] = record
    import numpy as np
    assert np.__version__=='1.26.4'
    rows,response = [],[]
    for index,rank in enumerate((35,100)):
        values = {}
        for case in CASES:
            t = native[case]['trials'][index]
            p = pin(t['output'],t['output_sha256'])
            nv = np.load(p,allow_pickle=False)
            assert hashlib.sha256(nv.tobytes()).hexdigest()==t['output_bytes_sha256']
            meta = hybrid['outputs'][str(rank)][case]['prediction']
            p = pin(meta['path'],meta['sha256'])
            hv = np.load(p,allow_pickle=False)
            assert hashlib.sha256(hv.tobytes()).hexdigest()==meta['raw_sha256']
            assert nv.dtype==hv.dtype==np.float32 and nv.shape==hv.shape==(16,1024)
            assert np.isfinite(nv).all() and np.isfinite(hv).all()
            difference = nv.astype(np.float64)-hv.astype(np.float64)
            values[case] = (nv.astype(np.float64),hv.astype(np.float64))
            rows.append(dict(rank=rank,case=case,
                maximum_absolute_by_layer=np.max(np.abs(difference),axis=1).tolist(),
                rms_by_layer=np.sqrt(np.mean(difference*difference,axis=1)).tolist()))
        for suffix,amplitude in (('025',.025),('050',.05)):
            plus,minus = values['plus'+suffix],values['minus'+suffix]
            nvec=(plus[0]-minus[0])/(2*amplitude)
            hvec=(plus[1]-minus[1])/(2*amplitude)
            diff=nvec-hvec
            response.append(dict(rank=rank,amplitude=amplitude,
                native_rms_by_layer=np.sqrt(np.mean(nvec*nvec,axis=1)).tolist(),
                hybrid_rms_by_layer=np.sqrt(np.mean(hvec*hvec,axis=1)).tolist(),
                discrepancy_rms_by_layer=np.sqrt(np.mean(diff*diff,axis=1)).tolist()))
    assert time.monotonic()-start<60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1_000_000_000
    for path,digest in pins.items():assert sha(path)==digest
    result=dict(schema='fixed_saved_native_p_layer_attribution_v1',completed=True,
        bridge_remains_failed=True,fit_admitted=False,gate_rejudged=False,
        arrays_only=True,model_calls=0,truth_accessed=False,weights_decoded=False,
        input_bindings=pins,rows=rows,responses=response,elapsed_s=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        finished_utc=datetime.now(timezone.utc).isoformat())
    out=HERE/'saved_layer_attribution_v1'
    out.mkdir(exist_ok=False)
    with (out/'result.json').open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
