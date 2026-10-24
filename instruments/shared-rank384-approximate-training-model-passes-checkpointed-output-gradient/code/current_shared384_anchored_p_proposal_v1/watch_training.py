"""Independent TRAIN20 watch:1500s whole,120s retired-call deadline,8GB,no swap."""
import hashlib
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'current_native_trajectory_adjoint_v1/supervise.py'
OWNER_SHA='648e5227795a8d2e5a91a1a421e0e54ffea1f76d6d6d65b28335178c8772025d'
TRAIN=(35,36,38,40,41,43,44,45,46,47,49,50,51,52,53,54,55,56,57,58)

PROGRESS='''    def progress():
        nonlocal last_progress,progress_count
        require(time.monotonic()-last_progress<120.,'No completed retired-call interval may reach120s')
        while progress_count<60:
            marker=output/f'call_{progress_count:03d}.retired.json'
            if not marker.exists():break
            require(marker.is_file() and not marker.is_symlink() and marker.stat().st_size<4096,
                'Bounded atomically published retirement marker')
            item=json.loads(marker.read_text());index=progress_count
            expected=output/f'call_{index:03d}.json'
            require(item['schema']=='anchored_train_call_retired_v1' and item['index']==index
                and item['rank']==_training_roster[index%20]
                and item['stage']==('baseline','training','terminal')[index//20]
                and item['call_graph_owners_retired'] and item['input_owners_retired']
                and item['shared_state_owners_may_persist']
                and item['record_path']==str(expected) and expected.stat().st_size<2*1024**2
                and sha(expected)==item['record_sha256'],'Actual ordered complete retired-call record')
            require(time.monotonic()-last_progress<120.,'Deadline checked before accepting progress')
            progress_count+=1;last_progress=time.monotonic()
            result['retired_call_count']=progress_count
'''


def derive():
    raw=OWNER.read_bytes()
    if OWNER.is_symlink() or hashlib.sha256(raw).hexdigest()!=OWNER_SHA:
        raise RuntimeError('Exact original independent resource supervisor')
    original=raw.decode();source=original;changes=[]
    for before,after in (
        ('CAP,RSS = 120.,8_000_000_000','CAP,RSS = 1500.,8_000_000_000'),
        ("choices=('native','adjoint')","choices=('training',)"),
        ('args = ap.parse_args(); began=time.monotonic(); last=began',
         'args = ap.parse_args(); began=_training_started; last=began; last_progress=began; progress_count=0'),
        ("'Whole120s, completed observations<=1s, tree<8GB, zero added swap'",
         "'Whole1500s, completed observations<=1s, tree<8GB, zero added swap'"),
        ("    try:\n        pins=json.loads(Path(args.pins).read_text())",
         PROGRESS+"    try:\n        pins=json.loads(Path(args.pins).read_text())"),
        ("and 'supervise.py' in pins,","and 'watch_training.py' in pins,"),
        ("        worker='run_'+args.phase+'.py'",
         "        require(sha(_training_owner)==_training_owner_sha,'Inherited supervisor at admission')\n"
         "        worker='run_'+args.phase+'.py'"),
        ("            observe()\n            if child.poll() is not None: break",
         "            observe();progress()\n            if child.poll() is not None: break"),
        ("records[0]['truth_accessed'] is False and records[0]['fits']==0",
         "records[0]['truth_accessed'] is True and records[0]['fits']==1 and progress_count==60"),
        ("records[0]['full_model_calls']==0",
         "records[0]['full_model_calls']==60 and records[0]['backwards']==20 and records[0]['optimizer_steps']==5"),
        ("'Actual target-free retired local qualifier, not a fitted estimator'",
         "'Actual60-call/20-backward/five-update TRAIN-only sole-payload dose'"),
        ("        if args.phase=='native':",
         "        require(len(records[0]['payloads'])==1 and records[0]['payloads'][0]['path']==str(output/'terminal_payload.npz')\n"
         "            and sha(output/'terminal_payload.npz')==records[0]['payloads'][0]['sha256']\n"
         "            ==records[0]['artifacts'][str(output/'terminal_payload.npz')]\n"
         "            and records[0]['native_score_measured'] is False and records[0]['release_ready'] is False,\n"
         "            'One actual terminal artifact, never a native-score or release qualification')\n"
         "        if args.phase=='native':"),
        ("        require(all(sha(HERE/k)==v for k,v in pins.items()),'Exact sources unchanged')",
         "        require(all(sha(HERE/k)==v for k,v in pins.items()),'Exact sources unchanged')\n"
         "        require(sha(_training_owner)==_training_owner_sha,'Inherited supervisor at finalization')\n"
         "        result['inherited_source_bindings']={str(_training_owner):_training_owner_sha}"),
    ):
        if source.count(before)!=1:raise RuntimeError('Exact prospective TRAIN stage/roster splice: '+before[:80])
        at=source.index(before);changes.append((at,before,after));source=source[:at]+after+source[at+len(before):]
    inverse=source
    for at,before,after in reversed(changes):
        if inverse[at:at+len(after)]!=after:raise RuntimeError('Exact whole resource supervisor inverse')
        inverse=inverse[:at]+before+inverse[at+len(after):]
    if inverse!=original:raise RuntimeError('All original RSS/swap/observation/kill/reap/tail checks preserved')
    return source


def main():
    began=time.monotonic()
    env=dict(__file__=str(Path(__file__).resolve()),__name__='_anchored_training_watch',
        _training_owner=OWNER,_training_owner_sha=OWNER_SHA,_training_roster=TRAIN,_training_started=began)
    exec(compile(derive(),str(Path(__file__).resolve())+':prospective-stage-inverse','exec'),env)
    return env['main']()


if __name__=='__main__':raise SystemExit(main())
