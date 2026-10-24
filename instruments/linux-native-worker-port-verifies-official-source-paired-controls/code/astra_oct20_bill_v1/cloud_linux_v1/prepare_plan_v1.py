"""Freeze a small Linux run plan from the existing curated transfer manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from common_linux_v1 import no_governor, path_under, sha, utc, verified, write_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--transfer-manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--verify-data', action='store_true', help='rehash all selected payload data on receiving host')
    p.add_argument('--candidate', action='append', default=[], metavar='LABEL=RELATIVE_PATH')
    a = p.parse_args()
    root = a.root.resolve(); transfer = a.transfer_manifest.resolve()
    m = json.loads(transfer.read_text())
    sources = {Path(x['path']).name: x for x in m['source']}
    tail = sources['tail_disable_fast_rescue_astra_v1.py']
    candidates = {}
    for label, name, append in [('bank337393', 'candidate_gram_pair_kclm_copy_astra_sci_v1.py', True),
                                ('bridge_micro', 'candidate_micro_kclm_copyall_nogov_astra_sci_v1.py', False)]:
        rec = dict(sources[name]); payload = verified(root, rec).read_bytes()
        if append:
            rec['append_tail'] = tail
            payload += b'\n' + verified(root, tail).read_bytes()
        no_governor(payload.decode())
        rec['effective_sha256'] = hashlib.sha256(payload).hexdigest()
        candidates[label] = rec
    for extra in a.candidate:
        label, rel = extra.split('=', 1)
        assert re.fullmatch(r'[a-zA-Z0-9_-]+', label) and label not in candidates
        f = path_under(root, rel); no_governor(f.read_text())
        candidates[label] = dict(path=rel, sha256=sha(f), bytes=f.stat().st_size, effective_sha256=sha(f))
    rows = {}
    for rec in m['data']:
        if a.verify_data:
            verified(root, rec)
        if rec['role'] == 'selected_weights':
            rows.setdefault(str(rec['rank']), {})['weights'] = rec
        else:
            match = re.fullmatch(r'mps_truth_rank(\d+)_([AB])32M\.npz', Path(rec['path']).name)
            assert match, rec
            rank, half = int(match[1]), match[2]
            rows.setdefault(str(rank), {})['truth_' + half] = dict(rec, key='mc_post')
    ranks = [r for group in m['groups'] for r in group]
    assert len(ranks) == len(set(ranks)) == 32 and set(map(int, rows)) == set(ranks)
    assert all(set(row) == {'weights', 'truth_A', 'truth_B'} for row in rows.values())
    sdk = {'client': [], 'server': []}
    for rec in m['sdk_reference_sources']:
        role = 'client' if rec['role'].endswith('_client') else 'server'
        rel = rec['path'].split('/site-packages/', 1)[1]
        sdk[role].append(dict(path=rel, sha256=rec['sha256']))
    clock = sources['setup_clock.py']; verified(root, clock)
    plan = dict(utc=utc(), kind='Linux paired science; no release/fire authority', fixture_mode='store32',
                transfer_manifest=dict(path=str(transfer.relative_to(root)), sha256=sha(transfer)),
                original_payload_bytes=m['size']['all_listed_bytes'], groups=m['groups'], first7=m['first7'],
                held25=m['held25'], rows=rows, candidates=candidates, packages=m['packages'], sdk_sources=sdk,
                setup_clock=clock, mac_reference=m['mac_reference'],
                data_verified_here=a.verify_data, budget=2**41, max_array_count=100000, max_array_bytes=4294967296)
    write_new(a.out, plan)
    print(json.dumps(dict(plan=str(a.out), sha256=sha(a.out), candidates={k:v['effective_sha256'] for k,v in candidates.items()})))


if __name__ == '__main__':
    main()
