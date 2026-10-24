"""Read-only status of the exact accepted SHARED Phase2 milestone submission.

Inert without --poll. One logical GET, pinned original client/POLL_RETRY,
explicit accepted id/result hash, no upload/forum/login/config write. Only
approved own-submission fields are saved in a new timestamped receipt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import sys

HERE = Path(__file__).resolve().parent
P2 = HERE.parent.parent
ENV = P2/'data_gate/.venv'
OUT = HERE/'release_v1'
CLIENT = ENV/'lib/python3.11/site-packages/whestbench/aicrowd_client.py'
CLIENT_SHA = '441cf36acb61266a960164de80fc66b21c5015773a898c76d1807b9c46bae6a6'
CANDIDATE_SHA = '4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3'
KEY_FILE = Path('/Users/matt/arc-white-box-estimation/key.txt')


def require(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(P2)
        and p.name != 'key.txt' and p.suffix in ('.py','.json'), 'regular_public_metadata')
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(path):
    p = Path(path)
    require(p.suffix == '.json' and 0 < p.stat().st_size < 16*1024**2, 'bounded_metadata')
    return json.loads(p.read_text())


def accepted(args):
    require(re.fullmatch('[0-9a-f]{64}',args.source_sha256 or '')
        and sha(__file__) == args.source_sha256, 'reviewed_poll_source')
    require(re.fullmatch('[0-9a-f]{64}',args.submit_sha256 or '')
        and sha(OUT/'submit_result.json') == args.submit_sha256, 'actual_accepted_submission_result')
    r = read(OUT/'submit_result.json')
    c,f = (read(OUT/name) for name in ('submit_completion.json','submit_finalization.json'))
    require(all(x.get('passed') is True and x.get('completed') is True for x in (r,c,f))
        and c['result_sha256'] == f['result_sha256'] == args.submit_sha256
        and f['completion_sha256'] == sha(OUT/'submit_completion.json')
        and r['schema'] == 'current_comp_shared_old384_release_v1'
        and r['phase'] == 'submit' and r['candidate_sha256'] == CANDIDATE_SHA
        and r['uploaded'] is True and r['official_score'] is False
        and r['submission_id'] == args.submission_id and type(args.submission_id) is int
        and args.submission_id > 0, 'exact_successfully_accepted_phase2_submission')
    return dict(result_sha256=args.submit_sha256,
        completion_sha256=sha(OUT/'submit_completion.json'),
        finalization_sha256=sha(OUT/'submit_finalization.json'))


def project(data, sid, key):
    require(isinstance(data,dict) and data.get('id') == sid
        and data.get('challenge_id') == 1174 and data.get('challenge_round_id') == 1429,
        'exact_own_phase2_status_identity')
    result = {name:data[name] for name in ('id','challenge_id','challenge_round_id')}
    for name in ('score','score_secondary'):
        v = data.get(name)
        if v is None or (type(v) in (int,float) and math.isfinite(v)):
            result[name] = v
    for name,limit in (('created_at',100),('grading_status_cd',64),('grading_message',2000)):
        v = data.get(name)
        if v is None or (isinstance(v,str) and len(v) <= limit and key not in v):
            result[name] = v
    require(key not in json.dumps(result,sort_keys=True), 'no_credential_in_public_projection')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poll',action='store_true')
    parser.add_argument('--submission-id',type=int)
    parser.add_argument('--source-sha256')
    parser.add_argument('--submit-sha256')
    args = parser.parse_args()
    if not args.poll:
        parser.print_help(); return 0
    key = None
    try:
        require(Path(sys.prefix).resolve() == ENV.resolve() and sys.flags.optimize == 0
            and sys.dont_write_bytecode and importlib.metadata.version('whestbench') == '0.16.0'
            and os.environ.get('RAILS_HOST','www.aicrowd.com') == 'www.aicrowd.com'
            and sha(CLIENT) == CLIENT_SHA, 'pinned_official_read_only_client')
        seals = accepted(args)
        # Importing whestbench.__init__ would pull in numerical dataset code.
        # This pinned standalone client has no relative package imports.
        spec = importlib.util.spec_from_file_location('shared_read_only_aicrowd_client',CLIENT)
        client_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = client_module
        spec.loader.exec_module(client_module)
        AIcrowdClient = client_module.AIcrowdClient
        import httpx
        require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'no_numerical_imports')
        def deadline(*unused):
            raise TimeoutError('bounded_read_only_poll')
        prior = signal.signal(signal.SIGALRM,deadline)
        signal.setitimer(signal.ITIMER_REAL,30.)
        try:
            key = KEY_FILE.read_text().strip()
            require(bool(key) and '\n' not in key and '\r' not in key, 'existing_key_unavailable')
            with httpx.Client(timeout=10.,follow_redirects=False) as http:
                client = AIcrowdClient(api_key=key,http=http)
                reply = client.get_submission_status(args.submission_id)
                result = project(reply.get('data',reply),args.submission_id,key)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0.)
            signal.signal(signal.SIGALRM,prior)
        now = datetime.now(timezone.utc)
        result.update(observed_by_utc=now.isoformat(),accepted_submission_seals=seals,
            source='Authenticated own-submission Phase2 status, read-only',
            source_sha256=args.source_sha256,client_sha256=CLIENT_SHA,
            candidate_sha256=CANDIDATE_SHA,per_mlp_counts_available=False,
            submission_attempted=False,forum_action=False)
        path = OUT/('official_status_'+now.strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        with path.open('x') as stream:
            json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        print(json.dumps({**result,'receipt':str(path),'receipt_sha256':sha(path)},allow_nan=False),flush=True)
        return 0
    except Exception as error:
        # Never format the raw exception, HTTP response, environment or key.
        print(json.dumps(dict(submission_id=args.submission_id,status_read_failed=True,
            error_type=type(error).__name__,no_submission_attempted=True)),flush=True)
        return 1
    finally:
        key = None


if __name__ == '__main__':
    raise SystemExit(main())
