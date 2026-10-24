"""Current physical model with ordinary differentiable CPU products.

This is an explicitly approximate proposal engine, not a participant or a
native derivative. Imports are inert until runtime/parameter/Card calls.
"""
import ast
import base64
import builtins
import hashlib
import math
import types
import zlib
from common import *

SOURCE = PARENT
PARENT_SHA = SOURCE_PARENT_SHA
PKEYS = BASE_PKEYS

FUNCTIONS = ('gaussian_raw', 'derivatives', '_pair_series_fused',
             '_nonlinear_pair_batch', '_nonlinear_first_gaussian', 'nonlinear_scalar',
             'features', 'final_scalar_features', 'gnn_step', 'silu',
             '_bdnh_coefficients', '_bd_update', '_edge_source',
             '_flms_star', '_flms_star_diagonal')


def extracted():
    assert sha(SOURCE) == PARENT_SHA
    text = SOURCE.read_text()
    tree = ast.parse(text)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            found[node.name] = node
    assert set(found) == set(FUNCTIONS)
    return ast.Module(body=[found[k] for k in FUNCTIONS], type_ignores=[])


def parameters(np):
    wanted = ('_GNN_LITERAL', '_BDNH_A_LITERAL', '_BD_COEFFICIENT_LITERAL', '_EDGE_LITERAL')
    literals = {}
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in wanted:
                assert key not in literals
                literals[key] = ast.literal_eval(node.value)
    assert set(literals) == set(wanted)
    packed = literals['_GNN_LITERAL']
    p = {k: np.frombuffer(zlib.decompress(base64.b64decode(blob, validate=True)),
                         dtype=np.float32).reshape(shape).copy()
         for k, (shape, blob) in packed.items()}
    a = np.asarray(literals['_BDNH_A_LITERAL'], dtype=np.float32)
    c = np.asarray(literals['_BD_COEFFICIENT_LITERAL'], dtype=np.float32)
    e = {k: np.asarray(v, dtype=np.float32) for k, v in literals['_EDGE_LITERAL'].items()}
    assert sum(p[k].size for k in PKEYS) == 45200 and a.shape == (64, 4) and c.shape == (15, 4)
    assert all(np.isfinite(v).all() for v in list(p.values()) + [a, c] + list(e.values()))
    return p, a, c, e


class Translate(ast.NodeTransformer):
    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == 'fnp'
                and call.func.attr == 'fill_diagonal'):
            assert len(call.args) == 2 and not call.keywords and isinstance(call.args[0], ast.Name)
            return ast.copy_location(ast.Assign(
                targets=[ast.Name(id=call.args[0].id, ctx=ast.Store())],
                value=ast.Call(func=ast.Name(id='_proposal_diagonal_write', ctx=ast.Load()), args=call.args, keywords=[])), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'astype':
                node.func.attr = 'to'
            elif node.func.attr == 'copy':
                assert not node.args and not node.keywords
                node.func.attr = 'clone'
            elif node.func.attr == 'transpose' and len(node.args) > 2:
                node.func.attr = 'permute'
        return node


def source_environment(api, cdf, edge, torch=None):
    tree = extracted()
    if torch is not None:
        tree = Translate().visit(tree)
    tree = ast.fix_missing_locations(tree)
    env = dict(fnp=api, flops=types.SimpleNamespace(stats=types.SimpleNamespace(norm=types.SimpleNamespace(cdf=cdf))),
               _EDGE_PARAMETERS=edge, _mm=lambda x, y: x @ y,
               str=lambda v: builtins.str(v).removeprefix('torch.'))
    if torch is not None:
        env['_proposal_diagonal_write'] = lambda x, v: torch.diagonal_scatter(
            x, torch.as_tensor(v, dtype=x.dtype, device=x.device).expand(x.shape[0]))
    exec(compile(tree, str(SOURCE) + ':explicit_offline_translation', 'exec'), env)
    env['translation_ast_sha256'] = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    return env


class TorchNP:
    def __init__(self, torch):
        self.t = torch
        self.float32, self.float64 = torch.float32, torch.float64

    def asarray(self, x, dtype=None):
        return self.t.as_tensor(x, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return self.t.zeros(shape, dtype=dtype or self.float64)

    def ones(self, shape, dtype=None):
        return self.t.ones(shape, dtype=dtype or self.float64)

    def arange(self, *a, dtype=None):
        return self.t.arange(*a, dtype=dtype)

    def sum(self, x, axis=None, dtype=None):
        return self.t.sum(x, dim=axis, dtype=dtype)

    def mean(self, x, axis=None):
        return self.t.mean(x, dim=axis)

    def stack(self, x, axis=0):
        return self.t.stack(x, dim=axis)

    def concatenate(self, x, axis=0):
        return self.t.cat(x, dim=axis)

    def cumprod(self, x, axis):
        return self.t.cumprod(x, dim=axis)

    def diag(self, x):
        return self.t.diag(x)

    def maximum(self, a, b):
        if not isinstance(a, self.t.Tensor):
            a = self.t.as_tensor(a, dtype=b.dtype, device=b.device)
        if not isinstance(b, self.t.Tensor):
            b = self.t.as_tensor(b, dtype=a.dtype, device=a.device)
        return self.t.maximum(a, b)

    def sqrt(self, x):
        return self.t.sqrt(x if isinstance(x, self.t.Tensor) else self.t.tensor(x, dtype=self.float64))

    def clip(self, x, lo, hi):
        return self.t.clamp(x, lo, hi)

    def __getattr__(self, name):
        assert name in ('exp', 'log', 'log1p', 'tanh', 'abs', 'outer', 'trace',
                        'zeros_like', 'ones_like', 'broadcast_to'), name
        return getattr(self.t, name)


def torch_payload(torch, np, trainable):
    p, a, c, e = parameters(np)
    p = {k: torch.tensor(v, dtype=torch.float32, requires_grad=trainable and k in PKEYS) for k, v in p.items()}
    return p, torch.tensor(a), torch.tensor(c), {k: torch.tensor(v) for k, v in e.items()}


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class Card:
    def __init__(self, torch, weights, p, a, c, edge):
        self.t = t = torch
        self.w = t.as_tensor(weights.copy(), dtype=t.float32)
        self.ws = self.w.to(t.float64)
        assert tuple(self.w.shape) == (16, 1024, 1024)
        self.P, self.A, self.cstar, self.edge = p, a, c, edge
        assert not a.requires_grad and not c.requires_grad and all(not x.requires_grad for x in edge.values())
        self.env = source_environment(TorchNP(t), lambda a: t.erfc(-a / math.sqrt(2)) / 2, edge, t)
        shared = {}
        exec(compile(SHARED.read_text(), str(SHARED), 'exec'), shared)
        # Match the native library's QR work precision; returned carrier is FP32.
        shared['_qr'] = lambda torch, y: torch.linalg.qr(y.to(torch.float64), mode='reduced')[0].to(y.dtype)
        self.shared = shared
        self.domain, self.selectors, self.states, self.visits = {}, {}, {}, {}
        self.call_layer = None
        self.slot = 0

    def diagonal(self, x, value):
        t = self.t
        return t.diagonal_scatter(x, t.as_tensor(value, dtype=x.dtype).expand(x.shape[0]))

    def observe(self, kind, layer, record):
        owner = getattr(self, kind)
        key = str(layer)
        if key in owner:
            assert owner[key] == record, ('checkpoint_replay', kind, key)
        owner[key] = record
        visit = kind + ':' + key
        self.visits[visit] = self.visits.get(visit, 0) + 1

    def variance(self, layer, v):
        t = self.t
        value = v.detach()
        r = dict(minimum=float(value.min()), finite=bool(t.isfinite(value).all()), sha256=tensor_sha(value))
        assert r['finite'] and r['minimum'] >= 1e-10, ('variance', layer, r)
        self.observe('domain', layer, r)

    def select(self, score):
        t = self.t
        count = min(1024, ((int((score.detach() >= .01).sum()) + 31) // 32) * 32)
        ids = t.argsort(-score.detach(), stable=True)[:count]
        alive = t.zeros_like(score).scatter(0, ids, t.ones(count, dtype=score.dtype))
        key = f'{self.call_layer}:{self.slot}'
        self.slot += 1
        assert ids.unique().numel() == count and int(alive.sum()) == count
        self.observe('selectors', key, dict(count=count, ids=tensor_sha(ids), mask=tensor_sha(alive)))
        return ids, alive

    def initial(self):
        t = self.t
        return (t.zeros(1024, dtype=t.float64), t.eye(1024, dtype=t.float64),
                t.empty((1024, 0)), t.empty(0), t.zeros((1024, 1024)),
                t.zeros(1024, dtype=t.float64), t.ones(1024, dtype=t.float64),
                t.zeros((), dtype=t.float64), t.zeros((1024, 64)),
                t.empty((1024, 0)), t.empty((0, 0)), t.empty((0, 0)), (), t.arange(1024))

    def step(self, layer, state):
        t, f = self.t, self.env
        self.call_layer, self.slot = layer, 0
        m, C, p, lam, star, b, previous_alive, radial, hstate, Q, F, core, stops, previous = state
        w, w64 = self.w[layer], self.ws[layer]
        mu = w64.T @ m
        pp, U, F, core, old_columns = self.shared['push'](
            t, layer, w, p, lam, previous, Q, F, core, stops, rank=384)
        self.observe('states', layer, dict(columns=int(pp.shape[1]), old_columns=old_columns,
            F_shape=list(F.shape), core_shape=list(core.shape), stops=list(stops)))
        bpre = w64.T @ b
        if layer == 15:
            pushed = C.to(t.float32) @ w
            vv = (w64 * pushed.to(t.float64)).sum(0)
            self.variance(layer, vv)
            p64 = pp.to(t.float64)
            k3 = ((p64 * p64 * p64) * lam.to(t.float64)).sum(1)
            k3 = k3 + f['_flms_star_diagonal'](w, star) + 3 * vv * bpre
            host, aux = f['nonlinear_scalar'](mu, vv, k3, radial * vv * vv)
            x = f['final_scalar_features'](layer, host, aux, w)
            delta, hs = f['gnn_step'](layer, x, aux['sigma'], w, hstate, self.P)
            return (host + delta.to(t.float64), C, p, lam, star, b, previous_alive,
                    radial, hs, Q, F, core, stops, previous)
        Cpre = ((w.T @ w) if layer == 0 else (w.T @ (C.to(t.float32) @ w))).to(t.float64)
        vv = t.diagonal(Cpre)
        self.variance(layer, vv)
        gate = t.erfc(-(mu / vv.clamp_min(1e-10).sqrt()) / math.sqrt(2)) / 2
        indices, alive = self.select(gate)
        live = alive[:, None] * alive[None, :]
        response_score = t.where(alive.bool(), t.minimum(gate, 1 - gate), -t.ones_like(gate))
        response_indices, response_mask = self.select(response_score)
        omitted = alive * (1 - response_mask)
        cp = t.zeros((1024, 1024), dtype=t.float64)
        if lam.numel():
            plane = self.shared['read'](t, pp, lam, response_indices, indices, U, F, old_columns).to(t.float64)
            cp = cp.index_put((response_indices[:, None], indices[None, :]), plane)
            p64 = pp.to(t.float64)
            cpdiag = ((p64 * p64 * p64) * lam.to(t.float64)).sum(1)
            cp = self.diagonal(cp, cpdiag)
        sp = t.zeros_like(Cpre) if layer == 0 else f['_flms_star'](w, star)
        k21 = cp + sp + vv[:, None] * bpre[None, :] + 2 * Cpre * bpre[:, None]
        k4 = radial * vv * vv
        k22 = radial * (t.outer(vv, vv) + 2 * Cpre * Cpre) / 3
        pair = f['_nonlinear_first_gaussian'] if layer == 0 else f['_nonlinear_pair_batch']
        host, C, target, new_radial, aux = pair(mu, Cpre, k21, k4, k22, 'covariance')
        if lam.numel():
            # Exact parent omitted_radial_source algebra, with current host.
            aa = (aux['gate'] - host * aux['pdf'] / aux['sigma']) * omitted
            beta = 2 * (aux['gaussian_mean'] - host * aux['gate']) * alive
            energy = (p64 * p64).T @ aa
            column = p64.T @ beta
            extra = 6 * ((lam.to(t.float64) * energy * column).sum() - (cpdiag * aa * beta).sum())
            new_radial = new_radial + extra / (t.trace(C) ** 2 + 2 * (C * C).sum())
        if 8 <= layer <= 14:
            C = C + f['_edge_source'](layer, host, k21, aux)
        x = f['features'](layer, host, C, k21, aux, w)
        delta, hs = f['gnn_step'](layer, x, aux['sigma'], w, hstate, self.P)
        delta = delta.to(t.float64)
        C = C - t.outer(host, delta) - t.outer(delta, host) - t.outer(delta, delta)
        m = host + delta
        g = aux['gate']
        d = aux['pdf'] / aux['sigma']
        birth = d * (1 + radial / 3)
        b = g * bpre + radial * d * vv / 6
        selected, births = self.select(t.minimum(g, 1 - g))
        inherited = g[:, None] ** 2 * g[None, :] * cp
        bp = self.diagonal(g[:, None] * Cpre, 0.)
        trace_post = t.diagonal(C)[:, None] * b[None, :] + 2 * C * b[:, None]
        beta = 2 * (aux['gaussian_mean'] - host * g)
        innovation = t.diagonal(target) - t.diagonal(inherited) - t.diagonal(trace_post)
        safe_birth = birth.clamp_min(1e-12)
        u = t.maximum(t.minimum(beta / safe_birth, 8 * vv), -8 * vv)
        disc = 4 * innovation / safe_birth - 3 * u * u
        gap = t.minimum(t.maximum(disc, .01 * vv * vv).sqrt(), 16 * vv)
        plus, minus = (u + gap) / 2, (u - gap) / 2
        coefficient = birth / gap
        born = bp * bp * (birth * births)[None, :] + (beta * births)[:, None] * bp.T
        born = self.diagonal(born, birth * births * (3 * u * u + gap * gap) / 4)
        c = self.cstar[layer] if layer == 0 else f['_bdnh_coefficients'](hs, self.A, self.cstar[layer])
        stored, increment = f['_bd_update'](bp, Cpre, k21, aux, d, alive, births, coefficient, plus, minus, c)
        born = born + increment
        cp_post = (inherited + born) * live
        positive = self.diagonal(stored, (plus * alive).to(t.float32))
        negative = self.diagonal(stored, (minus * alive).to(t.float32))
        p = t.cat((alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * pp),
                   positive[:, selected], negative[:, selected]), dim=1)
        lam = t.cat((lam, coefficient[selected].to(t.float32), -coefficient[selected].to(t.float32)))
        if old_columns:
            Q = alive.to(t.float32)[:, None] * (g.to(t.float32)[:, None] * U)
        stops = stops + (int(lam.numel()),)
        rest = target - cp_post - trace_post
        star = (3 * rest.T - 2 * t.diag(t.diagonal(rest))).to(t.float32)
        return m, C, p, lam, star, b, alive, new_radial, hs, Q, F, core, stops, indices

    def run(self, checkpointed):
        from torch.utils.checkpoint import checkpoint
        state = self.initial()
        outputs = []
        for layer in range(16):
            if checkpointed:
                state = checkpoint(lambda *args, layer=layer: self.step(layer, args),
                                   *state, use_reentrant=False)
            else:
                state = self.step(layer, state)
            outputs.append(state[0])
        return self.t.stack(outputs).clamp_min(0).to(self.t.float32)
