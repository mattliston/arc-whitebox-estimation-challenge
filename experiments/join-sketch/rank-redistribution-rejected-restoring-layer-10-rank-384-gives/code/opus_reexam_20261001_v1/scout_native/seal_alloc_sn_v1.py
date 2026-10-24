"""TAG scout-native seal_alloc_sn_v1 (2026-10-13: seal_block_v1.json is used up by s53k1) = the alloc command of engineer's
a62_modules/seal_judge_v1.py with the block file as an argument: the next n networks of <block json> that no earlier seal took (the shared
append-only ledger a62_modules/seal_alloc_v1.jsonl), recorded under <tag>, printed as a comma list.
usage: seal_alloc_sn_v1.py <tag> <n> <block json>"""
import json
import sys
import time
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / 'engineer/a62_modules/seal_alloc_v1.jsonl'
tag, n, block = sys.argv[1], int(sys.argv[2]), json.load(open(sys.argv[3]))['seeds']
used = {s for line in (LEDGER.read_text().splitlines() if LEDGER.exists() else []) for s in json.loads(line)['seeds']}
take = [s for s in block if s not in used][:n]
assert len(take) == n, 'the seal block is used up: ask the analyst for a new one'
with open(LEDGER, 'a') as f:
    f.write(json.dumps(dict(tag=tag, seeds=take, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))) + '\n')
print(','.join(map(str, take)))
