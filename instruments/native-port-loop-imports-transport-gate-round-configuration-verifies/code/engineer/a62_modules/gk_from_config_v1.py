"""TAG opus_engineer2 a62 gk_from_config_v1 (file written 2026-10-12T22:09:04Z; lead 21:4xZ task 3; trainer 21:48:23Z: gk11 ABSORBED at S53, every RAD
request from S53 on carries pbase gsk = gk4 = 1 through the round's config patch 'ncar', and the exporter carries no request field, so the
port applies gk natively): the native gk strengths of a RAD round, from its trainer/cert/RAD/S<N>/config.json.
  - no 'ncar' in the config patch: the default card (S51 / S52 requests: pbase gsk = gk4 = gbk = 0) -> 'GK NONE'
  - an 'ncar' list must be the default card's carrier list in the trainer's compact form (cert/RAD/ABSORB.json: n6y dfit 2; k31ly ages 1-3
    at k31 0 / 1 / 0 on leaves nkl1-3; pbase bsk 0.3402257055212658) plus pbase gsk / gk4 / gbk, the native _GSK / _GK4 / _GBK of
    a41_base/fragment_gsk_v1.py (applied by make_gsk_v1.py); anything else -> 'GK STOP <reason>', exit 3 (no native route known)
  - the round's resolved torch requests, when it has records (<results dir>, default trainer/run_RAD_S<N>_gen/results: the first
    a_RS<N>_base_r*.json, else any a_RS<N>*_r*.json 'ncar'), must be the default card's resolved list with these three strengths, else
    'GK STOP'; no record yet: config only
  - all three zero -> 'GK NONE'; else the values fragment: the absorbed delta's own a62_modules/fragment_gskv_<name>.py when it sets exactly
    these strengths (the port's edit is then the sealed stack's, byte for byte), else a new a62_modules/fragment_gskv_S<N>_v1.py;
    prints 'GK <values fragment> gsk <x> gk4 <y> gbk <z>'
usage: gk_from_config_v1.py <config.json> [<results dir> | none]"""
import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE.parents[1]
cfg_path = Path(sys.argv[1]).resolve()
cfg = json.loads(cfg_path.read_text())
N = int(re.fullmatch(r'S(\d+)', cfg['name']).group(1))
results = sys.argv[2] if len(sys.argv) > 2 else str(O / ('trainer/run_RAD_S%d_gen/results' % N))
STRENGTHS = ('gsk', 'gk4', 'gbk')
BASE_COMPACT = [{'module': 'n6y', 'dfit': 2}, {'module': 'k31ly', 'age': 1, 'k31': 0.0, 'leaf': 'nkl1'},
                {'module': 'k31ly', 'age': 2, 'k31': 1.0, 'leaf': 'nkl2'}, {'module': 'k31ly', 'age': 3, 'k31': 0.0, 'leaf': 'nkl3'},
                {'module': 'pbase', 'bsk': 0.3402257055212658}]
BASE_RESOLVED = json.loads(   # the resolved 'ncar' of every S51 / S52 request (a_RS51_base_r7100456.json, a_RS52_base_r7100712.json)
    '[{"module": "n6y", "arguments": {"k31": 1.0, "level": "read", "lo": 1, "hi": 14, "forms": 3, "dfit": 2, "dscale": 1.0, "ca": 0.75, '
    '"arest": false, "births": 1.0, "rows": 1.0, "diag": 1.0}}, {"module": "k31ly", "arguments": {"k31": 0.0, "age": 1, "lo": 1, "hi": 14, '
    '"forms": 0, "chaos4": true, "level": "read"}}, {"module": "k31ly", "arguments": {"k31": 1.0, "age": 2, "lo": 1, "hi": 14, "forms": 0, '
    '"chaos4": true, "level": "read"}}, {"module": "k31ly", "arguments": {"k31": 0.0, "age": 3, "lo": 1, "hi": 14, "forms": 0, "chaos4": true, '
    '"level": "read"}}, {"module": "pbase", "arguments": {"so2m": 0.0, "so2c": 0.0, "so2t": 0.0, "so2f": 0.0, "so2k": 1, "so2p": 123456, '
    '"so2td": 0, "bsk": 0.3402257055212658, "gsk": 0.0, "gk4": 0.0, "gbk": 0.0, "x1": 0.0, "x2": 0.0, "lo": 1, "hi": 14}}]')


def stop(why):
    print('GK STOP %s' % why)
    sys.exit(3)


ncar = cfg.get('patch', {}).get('ncar')
s = {k: 0.0 for k in STRENGTHS}
if ncar is not None:
    if not isinstance(ncar, list) or len(ncar) != len(BASE_COMPACT):
        stop('S%d config ncar is not the default carrier list (%d entries): %s' % (N, len(ncar) if isinstance(ncar, list) else -1, json.dumps(ncar)[:300]))
    for got, want in zip(ncar, BASE_COMPACT):
        got = dict(got)
        if got.get('module') == 'pbase':
            for k in STRENGTHS:
                if k in got:
                    s[k] = float(got.pop(k))
        if got != want:
            stop('S%d config ncar entry %s differs from the default card %s' % (N, json.dumps(got), json.dumps(want)))
recs = sorted(glob.glob('%s/a_RS%d_base_r*.json' % (results, N))) or sorted(glob.glob('%s/a_RS%d*_r*.json' % (results, N)))
if results != 'none' and recs:
    want = json.loads(json.dumps(BASE_RESOLVED))
    want[-1]['arguments'].update(s)
    if json.load(open(recs[0])).get('ncar') != want:
        stop('S%d resolved torch request %s carries another ncar than the default card with gsk %r gk4 %r gbk %r' % (N, os.path.basename(recs[0]), s['gsk'], s['gk4'], s['gbk']))
if not any(s.values()):
    print('GK NONE (S%d: %s; resolved records %s)' % (N, 'no ncar in the config patch' if ncar is None else 'gsk = gk4 = gbk = 0', os.path.basename(recs[0]) if results != 'none' and recs else 'not checked'))
    sys.exit(0)
want_vals = (s['gsk'], s['gk4'], s['gbk'])
values = None
for name in cfg.get('absorbed', []):
    p = HERE / ('fragment_gskv_%s.py' % name)
    if p.is_file():
        ns = {}
        exec(compile(p.read_text(), str(p), 'exec'), ns)
        if sorted(k for k in ns if not k.startswith('__')) == ['_GBK', '_GK4', '_GSK'] and (ns['_GSK'], ns['_GK4'], ns['_GBK']) == want_vals:
            values = p
            break
if values is None:
    values = HERE / ('fragment_gskv_S%d_v1.py' % N)
    body = ('# opus-engineer2 a62 gk_from_config_v1: S%d config.json patch.ncar pbase gsk %r gk4 %r gbk %r (absorbed %s)\n_GSK = %r\n_GK4 = %r\n_GBK = %r\n'
            % (N, *want_vals, ', '.join(cfg.get('absorbed', [])) or 'none named', *want_vals))
    if values.exists() and values.read_text() != body:
        stop('%s exists with other values' % values)
    if not values.exists():
        values.write_text(body)
print('GK %s gsk %r gk4 %r gbk %r (S%d; resolved records %s)' % (values, *want_vals, N, os.path.basename(recs[0]) if results != 'none' and recs else 'not checked'))
