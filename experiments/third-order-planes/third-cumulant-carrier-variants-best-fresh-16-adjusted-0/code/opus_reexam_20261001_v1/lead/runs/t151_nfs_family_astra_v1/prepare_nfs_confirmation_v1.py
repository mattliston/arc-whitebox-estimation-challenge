"""Prepare held confirmation files; optionally publish at a root-held boundary.

Default never touches live queue. Root must supply its saved family decision.
Publishing leaves the full T151 decision absent and v10 source untouched.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from table_nfs_family_v1 import HERE,O,P,R,FRESH,CONF,ARMS,sha,activation,screen


def request_map():
    found={}
    for directory in ('queue','claimed'):
        for path in (R/directory).glob('*.json'):
            try:d=json.loads(path.read_text())
            except FileNotFoundError:continue
            assert d['id'] not in found,'Duplicate request ID'
            found[d['id']]=(path,d)
    return found


def frozen(path,data):
    if path.exists():assert json.loads(path.read_text())==data,str(path)
    else:
        with path.open('x') as f:json.dump(data,f,allow_nan=False)


def boundary_check():
    assert (R/'STOP').exists(),'Root must hold the worker queue at a network boundary'
    import psutil
    for p in psutil.process_iter(['pid','name']):
        if 'python' not in (p.info['name'] or '').lower():continue
        try:args=p.cmdline()
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
        assert not (str(R) in args and any(Path(x).name in ('worker_ph153.py','worker_sc8.py') for x in args)), 'Queue worker still live'
    # With workers retired these missing results are stable; v10 cannot publish
    # T151 confirmation concurrently. v10 may continue unrelated bookkeeping.
    all_screen=[R/'results'/f'ph151{a}_r{s}.json' for a in ('sp','sm','dp','dm','kp') for s in FRESH]
    assert not all(p.exists() for p in all_screen),'Use already completed v10 queue instead'
    assert not (P/'t151_confirm_astra_v10.txt').exists(),'Full decision must remain absent'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root_decision',type=Path)
    ap.add_argument('--publish-after-root-boundary-hold',action='store_true')
    args=ap.parse_args();activation()
    decision=json.loads(args.root_decision.read_text())
    assert decision==screen(),'Root decision differs from immutable full-family table'
    assert decision['winner'] in ARMS
    winner=decision['winner'];priority=decision['priority']
    requests=request_map();source=requests[f'ph151{winner}_r{FRESH[0]}'][1]
    held=HERE/'held_confirmation';held.mkdir(exist_ok=True)
    prepared=[]
    for seed in CONF:
        rid=f'ph151x{winner}_r{seed}'
        req=copy.deepcopy(source);req.update(id=rid,rank=seed)
        name=(f'ph151_0prio_t151nfs_{rid}_astra_v1.json' if priority=='before_t152' else rid+'.json')
        target=held/name;frozen(target,req)
        if rid in requests:assert requests[rid][1]==req,'Existing canonical ID has different payload'
        prepared.append(dict(id=rid,seed=seed,held=str(target),queue_name=name,sha256=sha(target)))
    frozen(HERE/'HELD_CONFIRMATION_MANIFEST_v1.json',dict(root_decision_sha256=sha(args.root_decision),winner=winner,priority=priority,requests=prepared))
    if args.publish_after_root_boundary_hold:
        boundary_check();requests=request_map()
        for item in prepared:
            rid=item['id'];source=Path(item['held']);payload=json.loads(source.read_text());dest=R/'queue'/item['queue_name']
            if rid in requests:
                current,data=requests[rid];assert data==payload
                if current.parent==R/'queue' and current!=dest:
                    assert not dest.exists();current.rename(dest)
            else:
                assert not (R/'results'/(rid+'.json')).exists(),'Result missing preserved request'
                # Hard-link publishes complete immutable bytes; no partial JSON.
                assert not dest.exists();os.link(source,dest)
            # Request ID and payload stay exactly what v10 will later reconstruct.
        frozen(HERE/'PUBLISHED_CONFIRMATION_RECEIPT_v1.json',dict(winner=winner,priority=priority,requests=prepared,full_t151_decision_written=False))
    print(json.dumps(dict(winner=winner,priority=priority,prepared=16,published=args.publish_after_root_boundary_hold,supervisor_unchanged=True)))


if __name__=='__main__':main()
