#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-M2): server_mem.py + PEAK LOCATOR. Same deterministic live-array accounting (distinct base buffers behind
stored handles), plus, per window, WHERE the peak happened: the server op being stored at the new peak, the client call stack
published by census_e2_fine.py through a shared mmap file ($E2_SITE_MMAP), and the 25 largest live base buffers (bytes, shape, dtype)
at the peak (snapshot taken lazily at the first free after the last peak rise, i.e. exactly the peak state). Measurement only.
usage: python server_mem_v2.py --url ... --timeout ...   (env E2_MEM_OUT, E2_MEM_CTL, E2_MEM_WIN, E2_SITE_MMAP)"""
import os, sys, threading, time, json, heapq, mmap
import numpy as np
from flopscope_server import _array_store as S
from flopscope_server import _request_handler as RH
from flopscope_server.__main__ import main
live, bases, bobj, cur, peak, win = {}, {}, {}, [0], [0], [0]
lock = threading.Lock()
op_now = ['-']; seq = [0]
wpeak = dict(op='-', site='-', seq=0, pending=False, snap=None)
SITE = os.environ.get('E2_SITE_MMAP'); site_mm = [None]
def site_now():
    if not SITE:
        return '-'
    if site_mm[0] is None:
        try:
            fd = os.open(SITE, os.O_RDONLY); site_mm[0] = mmap.mmap(fd, 4096, prot=mmap.PROT_READ); os.close(fd)
        except OSError:
            return '-'
    return site_mm[0][:4096].split(b'\0', 1)[0].decode(errors='replace')
_disp = RH.RequestHandler._dispatch
def dispatch(self, request):
    op_now[0] = request.get('op', '-'); seq[0] += 1
    return _disp(self, request)
RH.RequestHandler._dispatch = dispatch
def _base(a):
    b = a
    while isinstance(getattr(b, 'base', None), np.ndarray):
        b = b.base
    return b
def snapshot():
    top = heapq.nlargest(25, bases.items(), key=lambda kv: kv[1][0])
    return [(round(nb / 2**20, 2), list(getattr(bobj.get(k), 'shape', ())), str(getattr(bobj.get(k), 'dtype', '?'))) for k, (nb, c) in top]
_put, _free, _clear = S.ArrayStore.put, S.ArrayStore.free, S.ArrayStore.clear
def put(self, arr):
    h = _put(self, arr)
    try:
        b = _base(arr); key, nb = id(b), int(b.nbytes)
    except Exception:
        b = arr; key, nb = id(arr), int(getattr(arr, 'nbytes', 0) or 0)
    with lock:
        live[h] = (key, b)
        if key in bases:
            bases[key][1] += 1
        else:
            bases[key] = [nb, 1]; bobj[key] = b; cur[0] += nb
            if cur[0] > peak[0]: peak[0] = cur[0]
            if cur[0] > win[0]:
                win[0] = cur[0]; wpeak.update(op=op_now[0], site=site_now(), seq=seq[0], pending=True)
    return h
def free(self, handles):
    with lock:
        if wpeak['pending']:
            wpeak['snap'] = snapshot(); wpeak['pending'] = False
        for h in handles:
            item = live.pop(h, None)
            if item is not None:
                key = item[0]; bases[key][1] -= 1
                if bases[key][1] == 0:
                    cur[0] -= bases.pop(key)[0]; bobj.pop(key, None)
    return _free(self, handles)
def clear(self):
    with lock:
        live.clear(); bases.clear(); bobj.clear(); cur[0] = 0
    return _clear(self)
S.ArrayStore.put, S.ArrayStore.free, S.ArrayStore.clear = put, free, clear
out = os.environ.get('E2_MEM_OUT')
ctl = os.environ.get('E2_MEM_CTL')
win_out = os.environ.get('E2_MEM_WIN')
def dump():
    last = None
    with open(out, 'w') as f, open(win_out or os.devnull, 'w') as g:
        while True:
            label = None
            if ctl:
                try:
                    label = open(ctl).read().strip()
                except OSError:
                    label = None
            with lock:
                if label != last:
                    if last is not None:
                        if wpeak['pending']:
                            wpeak['snap'] = snapshot(); wpeak['pending'] = False
                        g.write(json.dumps(dict(window=last, peak_live_MB=round(win[0] / 2**20, 1), live_at_end_MB=round(cur[0] / 2**20, 1),
                                                handles=len(live), peak_op=wpeak['op'], peak_seq=wpeak['seq'], peak_site=wpeak['site'],
                                                peak_top25=wpeak['snap'])) + '\n'); g.flush()
                    win[0] = cur[0]; last = label; wpeak.update(op='-', site='-', seq=0, pending=False, snap=None)
                f.write(json.dumps([round(time.time(), 2), round(cur[0] / 2**20, 1), round(peak[0] / 2**20, 1), len(live)]) + '\n')
            f.flush(); time.sleep(0.2)
if out:
    threading.Thread(target=dump, daemon=True).start()
main()
