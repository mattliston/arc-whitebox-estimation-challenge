"""TAG opus_engineer a41: the physicist's request string of a refit base -> the native base file of the builder (text only, nothing is run).
The physicist's runner (O/physicist/e16b_f4b_frozen_s3v.py, table t4_run_ora52.sh) names every base term by keywords of one request,
for example 'so2m=1,gsk=1,gk4=1,gbk=1.5,bk3a=1.6,bk3b=5'. This writes a41_base/bases/<name>.json = {"levers": [...]} for the builder's
lever BT@<name> (stack/build.py; stack/law_file_from_trainer.sh <hand-over> <2|3> same <name>) and prints the native price of the base.
  so2m=<c> [so2r=<r>]      second-order mean term inside the pair map, E relu^2 term at r (absent or negative: the same as c)   -> SO2m<c>[r<r>]
  so2t=<t> [so2td=1] [so2k=<0|1>] [so2p=<digits>]  second-order part of the third-cumulant target (td = its diagonal alone, k = highest power of rho, p = parts)  -> SO2m<c>t<t>[d][k<k>][p<digits>]
  so2f=<f>                 the mean term at the final layer, after the learned correction                                      -> SO2m<c>f<f>
  bsk=<x>                  birth weight with the skew-corrected kink density                                                   -> BSK<x>
  gsk=<s> gk4=<k> gbk=<b>  transport gate: skew, kurtosis, b-mode feed (one evaluation whatever the combination)               -> GSK<s>[k<k>][b<b>]
  bk3a=<x1> bk3b=<x2> [bk3s=1]  collective mode times the SPECIFIC third cumulant in the (3,1) plane, through the extra plane   -> BK3r<x1>,<x2>
Refused, with the reason: so2c (the covariance list is not written; it loses in the table), bk22 / bk4 / bkbb (not written; they lose in the
physicist's table), bk3s=0 (the non-specific form), tkbl and k31 (dials of the card: they go through the trainer's hand-over, not through a base
lever), nc6k22 (no effect in the table), so2lo / so2hi other than 1 / 15.
An existing base file of the same name must hold the same levers (nothing is overwritten).
usage: base_from_request.py <name> '<key=value,key=value,...>'"""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
# requests per network (14 layers) and bill in % of the file's, from the fragments' unit checks (RESULT.md section 28)
PRICE = dict(SO2m=(140, 0.007), BSK=(84, 0.0), GSK=(70, 0.0), BK3r=(150, 0.03))
# the target path by (kmax, parts; 'D' = the diagonal alone): requests a network and bill in % of the file's (unit_so2_v2.py 1024: one layer's call minus the
# stand-in's 16 requests and 2.0 n^2, 14 layers, bill 2.573e11); with the target path on, the mean term costs 70 requests more instead of 140
TARGET_PRICE = {(1, 'abqcde'): (658, 0.297), (0, 'abqc'): (434, 0.145), (1, 'abde'): (434, 0.186), (0, 'ab'): (308, 0.086), (1, 'D'): (196, 0.005), (0, 'D'): (196, 0.005)}
REFUSED = dict(so2c='the second-order covariance list is not written natively (about 420 requests and 0.2% of bill a network; it loses in the table)',
               bk22='the cross form in the (2,2) plane is not written (+124% in the loop, physicist 17:47Z)', bk4='the per-unit cross form is not written',
               bkbb='the form of second order in b is not written (coefficient about 0 against the truth planes)',
               tkbl='the b-form size is a dial of the card: it comes with the hand-over, not as a base lever', k31='the K31 row level is a dial of the card: it comes with the hand-over',
               nc6k22='K22 from the matrix diagonal changed nothing in the table')


def number(x):
    return ('%r' % float(x)).rstrip('0').rstrip('.') if float(x) != int(float(x)) else '%d' % int(float(x))


def levers_of(request):
    o = {}
    for item in request.split(','):
        if item.strip():
            k, v = item.split('=')
            o[k.strip()] = float(v)
    for k in o:
        assert k not in REFUSED or not o[k], '%s: %s' % (k, REFUSED[k])
        assert k in ('so2m', 'so2r', 'so2t', 'so2td', 'so2k', 'so2p', 'so2f', 'so2lo', 'so2hi', 'bsk', 'gsk', 'gk4', 'gbk', 'bk3a', 'bk3b', 'bk3s', 'x1', 'x2') or k in REFUSED, 'unknown keyword %r' % k
    for new, old in (('x1', 'bk3a'), ('x2', 'bk3b')):      # pbase's names of the cross form's two strengths
        if new in o:
            assert old not in o; o[old] = o.pop(new)
    assert o.get('so2lo', 1.0) == 1.0 and o.get('so2hi', 15.0) == 15.0, 'so2lo / so2hi other than 1 / 15 are not written'
    out = []
    if o.get('so2m') or o.get('so2t') or o.get('so2f'):
        r = o.get('so2r', -1.0)
        assert o.get('so2m') or r < 0, 'so2r without so2m is not a base term of the table'
        lever = 'SO2m' + number(o.get('so2m', 0.0)) + ('' if r < 0 or r == o.get('so2m', 0.0) else 'r' + number(r))
        if o.get('so2t'):
            k, p = int(o.get('so2k', 1)), str(int(o.get('so2p', 123456)))
            assert k in (0, 1), 'so2k = 2 (rho squared) is not written'
            lever += 't' + number(o['so2t']) + ('d' if o.get('so2td') else ('' if k == 1 else 'k0') + ('' if p == '123456' else 'p' + p))
        if o.get('so2f'):
            lever += 'f' + number(o['so2f'])
        out.append(lever)
    else:
        assert o.get('so2r', -1.0) < 0, 'so2r without so2m is not a base term of the table'
    if o.get('bsk'):
        out.append('BSK' + number(o['bsk']))
    if o.get('gsk') or o.get('gk4') or o.get('gbk'):
        out.append('GSK' + number(o.get('gsk', 0.0)) + ('k' + number(o['gk4']) if o.get('gk4') else '') + ('b' + number(o['gbk']) if o.get('gbk') else ''))
    if o.get('bk3a') or o.get('bk3b'):
        assert o.get('bk3s', 1.0) == 1.0, 'bk3s=0 (the whole K21 plane in place of the specific part) is not written; it loses in t4_ora51.log'
        out.append('BK3r' + number(o.get('bk3a', 0.0)) + ',' + number(o.get('bk3b', 0.0)))
    return out


if __name__ == '__main__':
    name, request = sys.argv[1], sys.argv[2]
    levers = levers_of(request)
    assert levers, 'no base term in the request'
    path = HERE/'bases'/(name + '.json')
    if path.exists():
        assert json.loads(path.read_text())['levers'] == levers, '%s exists with other levers: choose another name' % path
    else:
        path.write_text(json.dumps(dict(levers=levers, request=request), indent=1) + '\n')
    import re
    so2 = [re.fullmatch(r'SO2m([0-9.]+)(?:r([0-9.]+))?(?:t([0-9.]+)(d)?(?:k([01]))?(?:p([1-6]+))?)?(?:f([0-9.]+))?', x) for x in levers if x.startswith('SO2m')]
    requests = sum(PRICE[k][0] for k in PRICE if k != 'SO2m' and any(x.startswith(k) for x in levers))
    bill = sum(PRICE[k][1] for k in PRICE if k != 'SO2m' and any(x.startswith(k) for x in levers))
    for m in so2:
        if float(m.group(1)):
            requests += 70 if m.group(3) else PRICE['SO2m'][0]; bill += 0.0 if m.group(3) else PRICE['SO2m'][1]
        if m.group(3):
            kmax = int(m.group(5) or 1)
            used = 'D' if m.group(4) else ''.join(c for c in ''.join('abqcde'[int(ch) - 1] for ch in (m.group(6) or '123456')) if kmax or c not in 'de')
            price = TARGET_PRICE.get((kmax, used))
            assert price is not None, 'no measured price for kmax %d parts %s: run unit_so2_v2.py 1024 with that setting first' % (kmax, used)
            requests += price[0]; bill += price[1]
        if m.group(7):
            requests += 7
    print(json.dumps(dict(base=name, levers=levers, requests_per_network=requests, bill_pct=round(bill, 3), file=str(path))))
