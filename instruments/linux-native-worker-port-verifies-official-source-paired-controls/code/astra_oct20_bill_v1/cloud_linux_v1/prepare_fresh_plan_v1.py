"""Freeze selected already-exported fresh fixtures. Does not generate data or truths.

Input JSON is {"rows": [{"rank":7100100,"weights":{path,sha256,bytes},
"truth_A":{path,sha256,bytes,key},"truth_B":{path,sha256,bytes,key}}, ...]}.
Both halves may reference one frozen npz with distinct mc_post_A/mc_post_B keys.
Only explicitly selected files need be transferred as an addendum to the32 set.
"""
import argparse
import copy
import json
from pathlib import Path
from common_linux_v1 import load_plan, sha, utc, verified, write_new


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'base-plan', 'fixtures', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    plan = copy.deepcopy(load_plan(a.root, a.base_plan))
    selected = json.loads(a.fixtures.read_text())['rows']; rows = {}
    for row in selected:
        rank = int(row['rank']); assert rank >= 7100000 and str(rank) not in rows
        rows[str(rank)] = {key: row[key] for key in ('weights', 'truth_A', 'truth_B')}
        for key, rec in rows[str(rank)].items():
            verified(a.root, rec)
            if key.startswith('truth_'):
                assert rec['key'] in ('mc_post', 'mc_post_A', 'mc_post_B')
    assert rows
    plan.update(utc=utc(), fixture_mode='fresh_selected', rows=rows,
                groups=[[int(rank)] for rank in rows], first7=[], held25=[],
                parent_plan_sha256=sha(a.base_plan), fresh_fixture_manifest_sha256=sha(a.fixtures),
                fresh_protocol='one setup per selected seed; per-seed warm-up then scored prediction; fixture load stays metered; paired Linux bank required')
    write_new(a.out, plan)
    print(json.dumps(dict(plan=str(a.out), sha256=sha(a.out), seeds=list(map(int, rows)))))


if __name__ == '__main__':
    main()
