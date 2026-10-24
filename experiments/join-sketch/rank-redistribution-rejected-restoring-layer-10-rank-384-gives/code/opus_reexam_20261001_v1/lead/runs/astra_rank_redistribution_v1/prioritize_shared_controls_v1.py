"""HELD scheduling helper: preserve C request bytes/IDs; no model imports or results values.
prepare reorders only an idle, held weighted queue. watch stops that queue after
all16 atomic C result files exist; already claimed evaluations finish normally.
Do not rerun weighted prepare after this filename-only scheduling receipt.
"""
import argparse,json,os,subprocess,time
from pathlib import Path
from table_rank_redistribution_v3 import HERE,WEIGHTED,sha,prereg,identity

def write_new(p,data):
    with p.open('x') as f:json.dump(data,f,indent=2)
def stamp():return subprocess.check_output(['date','-u','+%Y%m%dT%H%M%SZ'],text=True).strip()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('prepare','watch'))
    ap.add_argument('--root-scheduling-grant',action='store_true')
    args=ap.parse_args();assert args.root_scheduling_grant
    d=prereg();assert identity()['passed'],'Both weighted and private zero-C smokes must pass'
    run=WEIGHTED/'run_screen';ids=[f'wfit_z_r{s}' for s in d['screen_seeds']]
    if args.action=='prepare':
        assert (run/'STOP').exists() and (run/'HELD_BY_ROOT').exists()
        assert not (run/'LAUNCH_LOCK').exists(),'All old weighted guards must retire first'
        assert not any((run/'claimed').glob('*.json')),'Only pristine unclaimed screen queue may be reordered'
        assert not any((run/'results').glob('*.json')) and not any((run/'results').glob('*.error'))
        m=json.loads((run/'requests_manifest_v1.json').read_text());assert len(m['requests'])==48
        assert m['prereg_sha256']==sha(WEIGHTED/'PREREG_v2.json')
        rows=[]
        for rid in ids:
            old=run/'queue'/(rid+'.json');new=run/'queue'/('000_sharedC_'+rid+'.json')
            q=json.loads(old.read_text());assert q['id']==rid and q['cutsim']['opts']['law_fit_blend']==0
            assert sha(old)==m['requests'][rid] and not new.exists()
            rows.append(dict(old=str(old),new=str(new),id=rid,sha256=sha(old)))
        receipt=HERE/'SHARED_C_QUEUE_PRIORITY_v1.json';assert not receipt.exists()
        # There are no active consumers, confirmed by the held queue and absent launcher lock.
        for row in rows:
            old,new=Path(row['old']),Path(row['new'])
            os.link(old,new);old.unlink();assert sha(new)==row['sha256']
        write_new(receipt,dict(created_utc=stamp(),scheduling_only=True,request_bytes_and_ids_unchanged=True,rows=rows))
        print(json.dumps(dict(reordered=16,scientific_values_read=0)))
    else:
        receipt=HERE/'SHARED_C_QUEUE_PRIORITY_v1.json';assert receipt.exists()
        marker=HERE/'SHARED_C_COUNT_WATCH_LOCK';marker.mkdir()
        try:
            while True:
                errors=[rid for rid in ids if (run/'results'/(rid+'.error')).exists()]
                count=sum((run/'results'/(rid+'.json')).exists() for rid in ids)
                if errors or count==16:
                    (run/'STOP').touch(exist_ok=True)
                    p=HERE/('SHARED_C_BOUNDARY_HOLD_'+stamp()+'_'+str(os.getpid())+'.json')
                    write_new(p,dict(complete_controls=count,errors=errors,scientific_values_read=0,
                                     own_weighted_queue_STOP=True,inflight_evaluations_finish=True))
                    print(json.dumps(dict(complete_controls=count,errors=errors,receipt=str(p))))
                    return 0 if not errors else 2
                time.sleep(0.25)
        finally:marker.rmdir()
if __name__=='__main__':raise SystemExit(main())
