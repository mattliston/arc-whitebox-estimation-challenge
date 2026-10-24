#!/usr/bin/env python3
"""TAG opus-lead fire_once: upload ONE qualified archive to the board, once, with no retry. Lead only.
usage: fire_once.py <tag> <archive.tar.gz> <description text>
Writes O/fire/<tag>/{attempt,response,receipt}.json (a tag can be used once). The API key is read from key.txt, handed to the
child process in its environment only, and never printed, logged or stored; outputs are redacted before they are written.
The child patches the whestbench client to a single HTTP attempt per request (no silent re-posts), as the September script did."""
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path('/Users/matt/arc-white-box-estimation'); HERE = Path(__file__).resolve().parent
tag, archive, description = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3]
out = HERE / tag
assert archive.is_file() and archive.name.endswith('.tar.gz'), archive
assert not out.exists(), f'{out} exists: this tag was already used; inspect it before any repeat'
out.mkdir(parents=True)
CHILD = ("from whestbench import aicrowd_client as api\nimport whestbench.cli as cli\n"
         "_original_request = api.AIcrowdClient._request\n"
         "_single_attempt = api.RetryPolicy(max_attempts=1, base_delay=0, max_delay=0, deadline_s=60)\n"
         "_post_counts = {'uploading the artifact': 0, 'creating your submission': 0}\n"
         "def request_once(self, method, url, *, policy=None, **kwargs):\n"
         "    if method.upper() == 'POST':\n"
         "        operation = kwargs.get('op')\n"
         "        assert operation in _post_counts, 'Unexpected POST operation'\n"
         "        _post_counts[operation] += 1\n"
         "        assert _post_counts[operation] == 1, 'POST already attempted; no retry'\n"
         "    return _original_request(self, method, url, policy=_single_attempt, **kwargs)\n"
         "api.AIcrowdClient._request = request_once\nraise SystemExit(cli.main())\n")
sha = hashlib.sha256(archive.read_bytes()).hexdigest()
command = [str(ROOT / '.venv/bin/python'), '-B', '-c', CHILD, 'submit', str(archive), '--challenge',
           'arc-white-box-estimation-challenge-2026', '--description', description, '--yes', '--json']
(out / 'attempt.json').write_text(json.dumps(dict(status='attempt started', utc=datetime.now(timezone.utc).isoformat(), tag=tag,
                                                  archive=str(archive), sha256=sha, bytes=archive.stat().st_size, description=description), indent=1))
key = (ROOT / 'key.txt').read_text().strip(); assert key, 'missing credential'
env = dict(os.environ, AICROWD_API_KEY=key, PYTHONDONTWRITEBYTECODE='1')
try:
    try:
        r = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)
        response = dict(returncode=r.returncode, stdout=r.stdout.replace(key, '[REDACTED]'), stderr=r.stderr.replace(key, '[REDACTED]'), timed_out=False)
    except subprocess.TimeoutExpired as err:
        txt = lambda v: (v.decode(errors='replace') if isinstance(v, bytes) else v or '').replace(key, '[REDACTED]')
        response = dict(returncode=None, stdout=txt(err.stdout), stderr=txt(err.stderr), timed_out=True)
finally:
    env.pop('AICROWD_API_KEY', None); del key, env
(out / 'response.json').write_text(json.dumps(response, indent=1))
assert not response['timed_out'] and response['returncode'] == 0, 'response kept in response.json; NO automatic retry'
payload = json.loads(response['stdout'])
from whestbench.aicrowd_client import extract_submission_id   # noqa: E402  (run with the repo venv)
sid = extract_submission_id(payload)
if sid is None:
    for value in payload.values():
        if isinstance(value, dict):
            sid = extract_submission_id(value)
            if sid is not None: break
assert sid is not None and int(sid) > 0, 'no submission id in the response; kept; do not repeat blindly'
receipt = dict(status='submitted; grading pending', submission_id=int(sid), tag=tag, sha256=sha,
               url=f'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{sid}', utc=datetime.now(timezone.utc).isoformat())
(out / 'receipt.json').write_text(json.dumps(receipt, indent=1)); print(json.dumps(receipt), flush=True)
