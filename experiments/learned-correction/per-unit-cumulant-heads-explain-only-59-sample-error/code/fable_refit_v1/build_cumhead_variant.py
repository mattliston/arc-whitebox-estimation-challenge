#!/usr/bin/env python3
"""TAG fable-refit (RF-8): native implementation of the per-unit cumulant head (refit_cumhead.py): from the previous layer's GNN state
h (1024x64 FP32) compute c3 = amp*tanh(h.w3+b3), c4 = amp*tanh(h.w4+b4), cv = vamp*tanh(h.wv+bv); at layers >= 8 the pair map
receives K21 with diagonal*(1+c3), k4*(1+c4), K22*sqrt((1+c4)_j(1+c4)_k) with diagonal k4, and Cpre with diagonal*(1+cv) (pair map
only; all later consumers keep the originals, exactly as the torch research card does); the final scalar map receives v*(1+cv),
k3*(1+c3), k4*(1+c4). Parameters come from a refit_cumhead.py params .pt (head dict + amp/vamp) and, optionally, the joint groups
are exported too (export_params.py semantics).  usage: build_cumhead_variant.py <base_cand.py> <params.pt> <out.py> [--heads-only]"""
import sys, ast, json
from pathlib import Path
import numpy as np, torch
base, pt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); heads_only = '--heads-only' in sys.argv
saved = torch.load(pt); head = {k: v.numpy() for k, v in saved['head'].items()}
amp = float(saved.get('amp', 0.3)); vamp = float(saved.get('vamp', 0.0)) if 'vamp' in saved else (0.003 if 'wv' in head and float(np.abs(head['wv']).sum()) > 0 else 0.0)
k21amp = float(saved.get('k21amp', 0.0)); from_layer = int(saved.get('from_layer', 8))
src = base.read_text()
def rep(s, old, new, count=1):
    assert s.count(old) == count, (old[:60], s.count(old)); return s.replace(old, new)
# 1) live _predict_binary: add the hook parameter (last definition)
i = src.rfind("def _predict_binary(mlp, hook=None, cov32=True, metric='covariance', collect_states=False, gate_floor=0.01, final_hook=None, coefficient_hook=None, covariance_hook=None):\n")
assert i > 0
head_line = "def _predict_binary(mlp, hook=None, cov32=True, metric='covariance', collect_states=False, gate_floor=0.01, final_hook=None, coefficient_hook=None, covariance_hook=None):\n"
src = src[:i] + "def _predict_binary(mlp, hook=None, cov32=True, metric='covariance', collect_states=False, gate_floor=0.01, final_hook=None, coefficient_hook=None, covariance_hook=None, cumulant_hook=None):\n" + src[i + len(head_line):]
# final branch (last occurrence)
F = "                m, aux = nonlinear_scalar(mu, v, k3, radial * v * v)\n"
k = src.rfind(F); assert k > i
src = src[:k] + ("                k4f = radial * v * v\n"
                 "                if cumulant_hook is not None:\n                    v, k3, k4f = cumulant_hook(l, 'final', v, k3, k4f)\n"
                 "                m, aux = nonlinear_scalar(mu, v, k3, k4f)\n") + src[k + len(F):]
# pair map (last occurrence)
Pm = "            m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n"
k = src.rfind(Pm); assert k > i
src = src[:k] + ("            if cumulant_hook is not None and l >= " + str(from_layer) + ":\n"
                 "                Cpre_h, K21_h, k4_h, k22_h = cumulant_hook(l, 'pair', Cpre, K21, k4, k22)\n"
                 "                m, C, target, radial_new, aux = pair_map(mu, Cpre_h, K21_h, k4_h, k22_h, metric)\n"
                 "                del Cpre_h, K21_h, k4_h, k22_h\n"
                 "            else:\n"
                 "                m, C, target, radial_new, aux = pair_map(mu, Cpre, K21, k4, k22, metric)\n") + src[k + len(Pm):]
# 2) the learned-compact wrapper: define the hook and pass it (last definition of _predict_learned_compact)
j = src.rfind("def _predict_learned_compact(mlp):\n"); assert j > 0
seg_end = src.index("\n\n\n", j); seg = src[j:seg_end]
seg = rep(seg, "    def message_covariance_hook(l,m,K21,aux,w):\n",
          "    _ch = _cum_head_parameters()\n"
          "    def cumulant_hook(l, mode, *args):\n"
          "        h = state\n"
          "        c3 = (_ch['amp'] * fnp.tanh(h @ _ch['w3'] + _ch['b3'])).astype(fnp.float64)\n"
          "        c4 = (_ch['amp'] * fnp.tanh(h @ _ch['w4'] + _ch['b4'])).astype(fnp.float64)\n"
          "        cv = (_ch['vamp'] * fnp.tanh(h @ _ch['wv'] + _ch['bv'])).astype(fnp.float64) if _ch['vamp'] else None\n"
          "        c21 = (_ch['k21amp'] * fnp.tanh(h @ _ch['w21'] + _ch['b21'])).astype(fnp.float64) if _ch['k21amp'] else None\n"
          "        if mode == 'final':\n"
          "            v, k3, k4 = args\n"
          "            if cv is not None: v = v * (1 + cv)\n"
          "            return v, k3 * (1 + c3), k4 * (1 + c4)\n"
          "        Cpre, K21, k4, k22 = args\n"
          "        d21 = fnp.diag(K21).copy()\n"
          "        if c21 is not None:\n"
          "            s21 = fnp.sqrt(1 + c21); K21 = K21 * fnp.outer(s21, s21)\n"
          "        else:\n"
          "            K21 = K21.copy()\n"
          "        fnp.fill_diagonal(K21, d21 * (1 + c3))\n"
          "        k4 = k4 * (1 + c4)\n"
          "        s4 = fnp.sqrt(1 + c4); k22 = k22 * fnp.outer(s4, s4); fnp.fill_diagonal(k22, k4)\n"
          "        if cv is not None:\n"
          "            Cpre = Cpre.copy(); fnp.fill_diagonal(Cpre, fnp.diag(Cpre) * (1 + cv))\n"
          "        return Cpre, K21, k4, k22\n"
          "    def message_covariance_hook(l,m,K21,aux,w):\n")
seg = rep(seg, "                           covariance_hook=message_covariance_hook if _MESSAGE_EDGE_ENABLED else None)",
          "                           covariance_hook=message_covariance_hook if _MESSAGE_EDGE_ENABLED else None,\n                           cumulant_hook=cumulant_hook)")
src = src[:j] + seg + src[seg_end:]
lit = dict(w3=head['w3'].astype(float).tolist(), b3=float(head['b3']), w4=head['w4'].astype(float).tolist(), b4=float(head['b4']),
           wv=head.get('wv', np.zeros(64)).astype(float).tolist(), bv=float(head.get('bv', 0.0)), amp=amp, vamp=vamp,
           w21=head.get('w21', np.zeros(64)).astype(float).tolist(), b21=float(head.get('b21', 0.0)), k21amp=k21amp)
src += ("\n\n_CUM_HEAD_LITERAL = " + repr(lit) + "\n_CUM_HEAD_PARAMETERS = None\n"
        "def _cum_head_parameters():\n    global _CUM_HEAD_PARAMETERS\n    if _CUM_HEAD_PARAMETERS is None:\n"
        "        L = _CUM_HEAD_LITERAL\n"
        "        _CUM_HEAD_PARAMETERS = dict(w3=fnp.asarray(L['w3'], dtype=fnp.float32), b3=float(L['b3']), w4=fnp.asarray(L['w4'], dtype=fnp.float32), b4=float(L['b4']),\n"
        "                                    wv=fnp.asarray(L['wv'], dtype=fnp.float32), bv=float(L['bv']), amp=float(L['amp']), vamp=float(L['vamp']),\n"
        "                                    w21=fnp.asarray(L['w21'], dtype=fnp.float32), b21=float(L['b21']), k21amp=float(L['k21amp']))\n"
        "    return _CUM_HEAD_PARAMETERS\n")
out.write_text(src); ast.parse(src)
if not heads_only and 'p' in saved:
    import subprocess; subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'export_params.py'), str(out), str(pt), str(out)], check=True)
print(json.dumps(dict(out=str(out), amp=amp, vamp=vamp, bytes=len(out.read_text()))))
