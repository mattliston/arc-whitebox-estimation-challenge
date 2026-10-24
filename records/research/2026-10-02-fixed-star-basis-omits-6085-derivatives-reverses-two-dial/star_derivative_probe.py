"""TAG opus-trainer (2026-10-02): is the ATTACHED rank-32 star basis differentiated correctly, and over which step size is its
derivative valid? One forward run of the run M card captures the star A carried out of a few layers together with the next
layer's weights. For each, with a fixed random read-out G:  L(A) = sum(G * star_push(w, A))  (star_push = the engine's low-rank
star with its basis from the range finder). Directional derivatives of L along a few directions D (unit Frobenius norm times
the norm of A): autograd with the basis detached, autograd with the basis attached, and central differences of the float32
forward function for relative steps 1e-4 .. 1e-1. Directions: random Gaussian, random rank one, and b v^T-shaped (the shape a
change of the carried b or of F4's anchor gives the star: left factor random, right factor the diagonal of the carried covariance).
usage: TRAINER_THREADS=3 ... TRAINER_PL=1 TRAINER_CARD=M python -B star_derivative_probe.py <rank> [layers, default 4,8,12]"""
import json
import sys

import otr_engine as E

torch = E.torch
rank = int(sys.argv[1])
LAYERS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [4, 8, 12]
captured = {}
install = E.M.install


def with_capture(cd):
    install(cd)
    out0 = cd.hooks['state_out']
    def state_out(l, out, ctx):
        out = out0(l, out, ctx)
        if l in LAYERS:
            captured[l] = dict(A=out[4].detach().clone(), v=torch.diagonal(out[1]).detach().clone(), w=cd.w[l + 1].detach().clone())
        return out
    cd.hooks['state_out'] = state_out
    captured['star'] = cd.env['_flms_star']


E.M.install = with_capture
E.PL.set_values({})
record = E.run_network(rank, 'eval')
E.M.install = install
star = captured.pop('star')
torch.manual_seed(1)
print(json.dumps(dict(rank=rank, mse=record['mse'])), flush=True)
for l, cap in captured.items():
    A, w, v = cap['A'], cap['w'], cap['v'].to(torch.float32)
    G = torch.randn(1024, 1024, dtype=torch.float64)
    sv = torch.linalg.svdvals(A.double())
    def L(X):
        return (star(w, X) * G).sum()
    grads = {}
    for attach in (False, True):
        E.STAR_ATTACH = attach
        X = A.clone().requires_grad_(True)
        with torch.enable_grad():
            L(X).backward()
        grads[attach] = X.grad.double()
    E.STAR_ATTACH = False
    norm = A.double().norm()
    directions = dict(gaussian=torch.randn(1024, 1024), rank_one=torch.outer(torch.randn(1024), torch.randn(1024)),
                      b_times_v=torch.outer(torch.randn(1024), v), the_star_itself=A.clone())
    out = dict(layer=l, star_norm=float(norm), singular_values={k: float(sv[k - 1] / sv[0]) for k in (2, 8, 16, 32, 33, 48, 64, 128)},
               energy_in_top32=float((sv[:32] ** 2).sum() / (sv ** 2).sum()), directions={})
    for name, D in directions.items():
        D = (D / D.double().norm() * norm).to(torch.float32)      # a relative step t moves the star by t times its own norm
        row = dict(detached=float((grads[False] * D.double()).sum()), attached=float((grads[True] * D.double()).sum()), central={})
        with torch.no_grad():
            for step in (1e-4, 1e-3, 1e-2, 3e-2, 1e-1):
                row['central']['%g' % step] = float((L(A + step * D) - L(A - step * D)) / (2 * step))
        out['directions'][name] = row
    print(json.dumps(out), flush=True)
