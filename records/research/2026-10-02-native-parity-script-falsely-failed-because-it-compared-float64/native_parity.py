"""TAG opus-release: native parity of one file against a finished boundary session of the SAME file: the networks of the
session are run natively (OpenBLAS, in process, fresh process, same order, setup() first; stage_b_native40.native, under the
guard) and must give byte-identical predictions and the same bill as the session's networks. What stage D does for its first
two networks, as a separate step for a file whose stage D was run on a literal-only twin.
usage: native_parity.py <candidate.py> <session dir (rows.jsonl, pred_<rank>.bin)> [--no-gate]   -> prints one line per network and PARITY PASS / FAIL"""
import json
from pathlib import Path
import sys

import numpy as np

from common import sha
from stage_b_native40 import native

candidate, session = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
manifest = json.loads((session/'manifest.json').read_text())
assert manifest['sha256'] == sha(candidate), 'the session is not a session of this file'
rows = [json.loads(x) for x in (session/'rows.jsonl').read_text().splitlines()]
ranks = [r['rank'] for r in rows]
native_rows, where, stored = native(candidate, ranks, '--no-gate' not in sys.argv)
ok = True
for r, n in zip(rows, native_rows):
    # the comparison of stage D (stage_d_serving.py): the boundary's float32 bytes, widened, against the native float64 array
    rpc = np.frombuffer((session/f"pred_{r['rank']}.bin").read_bytes(), dtype=np.float32).reshape(16, 1024)
    same = rpc.astype(np.float64).tobytes() == np.load(where/f"pred_{r['rank']}.npy").tobytes()
    bill = n['flops'] == r['flops']
    ok = ok and same and bill
    print(f"network {r['rank']}: bytes identical {same}; bill native {n['flops']:,} / boundary {r['flops']:,} ({'equal' if bill else 'DIFFERENT'})")
print(f"[native parity] {'PASS' if ok else 'FAIL'}: {candidate.parent.name}/{candidate.name} (sha {sha(candidate)[:12]}) against {session} "
      f"({'stored native run' if stored else 'native run'} {where})")
sys.exit(0 if ok else 1)
