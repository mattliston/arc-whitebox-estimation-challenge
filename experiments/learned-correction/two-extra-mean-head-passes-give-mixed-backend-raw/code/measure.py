"""Full40 sealed surrogate magnitude screen, never a candidate score."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
BASE, P2 = HERE.parent, HERE.parent.parent
Q = BASE/'current_p1_terminal80_requalification_v1'
P1 = BASE/'current_d50_native_objective_p1_v1'
PINS = {
    Q/'inference_v1/result.json': '9cfee94622c238a8fc0dc6cf7454e28ec54b73fb4998697d30182343fc405bcf',
    Q/'inference_v1/completion.json': '5e330c62e6431adb84a726ad67d5d65c3b2413c71bf01079f95455feb1516881',
    Q/'inference_watch_v1/result.json': '5f1d0e310107cefe089d1be4509807492408cbdf0010f29aa28582afa5217701',
    Q/'inference_watch_v1/completion.json': 'f406fd38617f0aa1ce61e4e1fe7bcfca4d752081b2cdda19e8970dc8354e791b',
    P1/'native2_terminal_v1/result.json': '7a1f37d80a7ed32850b10047408d85eed050220b4a6271dde90bdd21c07062e3',
    P1/'native2_terminal_v1/completion.json': 'b73a957bf78e45da0db2741cea0b33a2cd98317726edaaa0954eaece054e5957',
    P1/'score2_v1/result.json': '4f60a8def29f1a9326d68d5c25bbe79b3f26289af8cf46de81609fb91cf055e5',
    P1/'score2_v1/completion.json': 'c72831702f18a70a4bc40a8d88a655879be3e816e3edd3351cce9ea75e8cd587',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    start = time.monotonic()
    out = HERE/'screen_v1'
    out.mkdir(exist_ok=False)
    bindings = {str(Path(__file__).resolve()): sha(__file__),
                str(HERE/'PROTOCOL.md'): sha(HERE/'PROTOCOL.md')}
    result = dict(passed=False, official_score=False, adjusted_score_measured=False,
                  native_score_measured=False, estimator_executed=False, refit=False,
                  checkpoint_selection=False, old_R2_remains_failed=True,
                  truth_accessed=False, comparison='terminal80 MPS versus P1 native; same fixed1e9 means',
                  started_utc=datetime.now(timezone.utc).isoformat())
    def guard(*_):
        check(time.monotonic()-start < 120, '120s cap')
        check(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 1_000_000_000, '1GB RSS cap')
    signal.signal(signal.SIGALRM, guard)
    signal.setitimer(signal.ITIMER_REAL, .2, .2)
    def pin(p,h):
        p=Path(p).resolve()
        check(p.is_relative_to(P2) and p.stat().st_size<10_000_000 and sha(p)==h, 'input binding '+str(p))
        bindings[str(p)]=h
        return p
    try:
        check(os.getpriority(os.PRIO_PROCESS,0)>=10 and sys.flags.optimize==0, 'nice10/not optimized')
        check(Path(sys.prefix).resolve()==(P2/'data_gate/.venv').resolve(), 'OFFPY owner')
        check(all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')), 'single thread environment')
        objs={p:json.loads(pin(p,h).read_text()) for p,h in PINS.items()}
        for p,d in objs.items():
            check(d['passed'], 'passed prerequisite')
            if p.name=='completion.json':
                check(d['result_sha256']==PINS[p.with_name('result.json')], 'completion binding')
        q=objs[Q/'inference_v1/result.json']; p=objs[P1/'native2_terminal_v1/result.json']
        score=objs[P1/'score2_v1/result.json']
        check(all(q.get(k) is True for k in ('all40_outputs_sealed','full_P_unchanged',
              'inference_integrity_passed','repeat37_byte_exact','old_R2_remains_failed','hashes_unchanged')), 'inference qualification')
        check(q['truth_accessed'] is False and len(q['validation_outputs'])==40, 'sealed target blind full40')
        check(p['truth_accessed'] is False and len(p['trials'])==41 and len(score['records'])==40, 'P1 native40')
        paths=[]
        for i,r in enumerate(score['records']):
            rank=r['rank']; t=p['trials'][i]; v=q['validation_outputs'][str(rank)]
            check(t['call']==r['call']==i and t['rank']==rank, 'same rank frame')
            paths.append((pin(v['path'],v['sha256']), pin(t['output'],t['output_sha256'])))
        qr=Q/'inference_v1/graph_0040_served.npy'
        pin(qr,q['artifacts'][str(qr)])
        pr=pin(p['trials'][40]['output'],p['trials'][40]['output_sha256'])
        check(qr.read_bytes()==paths[0][0].read_bytes() and pr.read_bytes()==paths[0][1].read_bytes(), 'both exact repeats')
        import numpy as np
        check(np.__version__=='2.4.6' and not any(x in sys.modules for x in ('torch','flopscope')), 'offline numeric owner')
        values=[]
        for pair in paths:
            a=[np.load(f,allow_pickle=False) for f in pair]
            check(all(v.dtype==np.float32 and v.shape==(16,1024) and np.isfinite(v).all() and (v>=0).all() for v in a), 'served format/domain')
            values.append(a)
        (out/'predictions_checked_before_truth.json').write_text(json.dumps(dict(
            input_bindings=bindings, all40_checked=True, repeat_excluded=True,
            truth_accessed=False),sort_keys=True)+'\n')
        result['all_predictions_verified_before_truth']=True
        rows=[]
        for r,(term,baseline) in zip(score['records'],values):
            guard()
            truth=P2/'public_g22_operator_features_v1/truthpost'/f"truthpost_rank{r['rank']:04d}.npz"
            pin(truth,score['truth_bindings'][str(truth)])
            result['truth_accessed']=True
            with np.load(truth,allow_pickle=False) as src:
                target=src['mc_post'].astype(np.float64)
            check(target.shape==(16,1024) and np.isfinite(target).all(), 'same finite target')
            em=[np.mean((v.astype(np.float64)-target)**2,axis=1) for v in (term,baseline)]
            check(em[1].tolist()==r['per_role']['terminal']['layer_mse'], 'exact P1 score replay')
            rows.append(dict(rank=r['rank'], terminal80_surrogate_raw=float(em[0][-1]),
                             p1_native_raw=float(em[1][-1]), terminal80_layer_mse=em[0].tolist()))
        a=math.fsum(r['terminal80_surrogate_raw'] for r in rows)/40
        b=math.fsum(r['p1_native_raw'] for r in rows)/40
        wins=sum(r['terminal80_surrogate_raw']<r['p1_native_raw'] for r in rows)
        for path,h in bindings.items():check(sha(path)==h,'input unchanged')
        guard()
        result.update(passed=True,records=rows,mean_terminal80_surrogate_raw=a,
                      mean_p1_native_raw=b,raw_ratio=a/b,raw_wins=wins,
                      native_stage_reconsideration_gate_passed=a/b<=.90 and wins>=24,
                      hashes_unchanged=True,smaller_components_preserved=True)
    except Exception as e:
        result['error']=f'{type(e).__name__}: {e}'
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result.update(input_bindings=bindings,elapsed_s=time.monotonic()-start,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        (out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
        (out/'completion.json').write_text(json.dumps(dict(passed=result['passed'],
            result_sha256=sha(out/'result.json')),sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_bindings','records')},indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
