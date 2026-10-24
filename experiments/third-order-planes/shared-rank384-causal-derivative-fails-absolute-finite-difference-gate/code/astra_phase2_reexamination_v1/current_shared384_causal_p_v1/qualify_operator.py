"""One fixed target-free shared-operator algebra/gradient qualification."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
import weakref

HERE = Path(__file__).resolve().parent
LIMITS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    with Path(path).open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pins', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    began = time.monotonic()
    output = Path(args.output).resolve()
    require(output.parent == HERE and not output.exists(), 'Fresh local output only')
    output.mkdir()
    result = dict(passed=False, completed=False, truth_accessed=False,
                  physical_networks=0, fits=0, native_score_measured=False, cases=[])
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('120s unit')))
    signal.alarm(120)
    try:
        pins = json.loads(Path(args.pins).read_text())
        require(set(pins) == {'shared_operator.py', 'qualify_operator.py',
                             'watch_operator.py', 'UNIT_PROTOCOL.md', 'OPERATOR_DESIGN.md'},
                'Exactly the five reviewed unit owners')
        require(all(sha(HERE / name) == digest for name, digest in pins.items()), 'Source pins')
        require(not {'numpy', 'torch', 'flopscope', 'threadpoolctl'} & set(sys.modules),
                'Inert source admission')
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10 and
                all(os.environ.get(key) == '1' for key in LIMITS), 'Nice10 and six BLAS limits')
        spec = importlib.util.spec_from_file_location('tiny_shared_operator', HERE / 'shared_operator.py')
        op = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(op)
        require(not {'numpy', 'torch', 'flopscope', 'threadpoolctl'} & set(sys.modules),
                'Pure operator import stays inert')
        import torch
        from torch.utils.checkpoint import checkpoint
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result['runtime'] = dict(torch=torch.__version__, device='cpu', precision='float64')
        generator = torch.Generator(device='cpu').manual_seed(202609131135)
        def rand(shape):
            return torch.randn(shape, generator=generator, dtype=torch.float64) * .2
        n, rank = 12, 3
        previous = torch.tensor([9, 0, 7, 2, 11, 5, 4, 8, 1, 6])
        active = torch.zeros(n, dtype=torch.float64).scatter(0, previous, 1.)
        response = torch.tensor([10, 1, 7, 3, 9])
        live = torch.tensor([11, 4, 0, 6, 2, 7, 5])
        def fixture(layer):
            stops = tuple(4 * (j + 1) for j in range(layer))
            columns = 4 * layer
            w = rand((n, n))
            p = rand((n, columns)) * active[:, None]
            pair = torch.linspace(.15, .35, columns // 2, dtype=torch.float64)
            lam = torch.cat(tuple(torch.stack((pair[2*j], pair[2*j+1],
                                              -pair[2*j], -pair[2*j+1]))
                                  for j in range(layer)))
            if layer < 6:
                Q, F, core = torch.empty((n, 0), dtype=torch.float64), \
                    torch.empty((0, 0), dtype=torch.float64), torch.empty((0, 0), dtype=torch.float64)
            else:
                old = stops[9] if layer == 15 else stops[layer - 6]
                Q = torch.linalg.qr(rand((n, rank)), mode='reduced')[0] * active[:, None]
                F = rand((rank, old))
                core = (F * lam[:old].square()) @ F.T
                p = torch.cat((Q @ F, p[:, old:]), dim=1)
            return (w, p, lam, Q, F, core), stops
        def byte_hash(x):
            return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()
        def reference(layer, operands, stops):
            w, p, lam, Q, F, core = operands
            if layer < 5:
                return w.T @ p, None, F, core, 0
            if layer < 15:
                start = 0 if layer == 5 else stops[layer - 6]
                stop = stops[layer - 5]
                J = p[previous, start:stop]
                G = (J * lam[start:stop].square()) @ J.T
                if start:
                    G = G + Q[previous] @ core @ Q[previous].T
                basis, triangular = torch.linalg.qr(G @ w[previous, :rank], mode='reduced')
                diagonal = triangular.diagonal().abs()
                require(float(diagonal.min()/diagonal.max()) > 1e-6,
                        'Full-column-rank well-conditioned fixed tiny QR')
                projection = basis @ basis.T
                old_full = p[previous, :stop]
                projected = projection @ old_full
                all_active = torch.cat((projected, p[previous, stop:]), dim=1)
                pp = w[previous].T @ all_active
                return pp, w[previous].T @ basis, basis.T @ old_full, basis.T @ G @ basis, stop
            old = int(F.shape[1])
            U = w[previous].T @ Q[previous]
            pp = torch.cat((U @ F, w[previous].T @ p[previous, old:]), dim=1)
            return pp, U, F, core, old
        def invoke(layer, operands, stops):
            w, p, lam, Q, F, core = operands
            return op.push(torch, layer, w, p, lam, previous, Q, F, core, stops, rank=rank)
        def err(x, y):
            if not x.numel():
                return dict(max_abs=0., relative_l2=0.)
            return dict(max_abs=float((x-y).abs().max()),
                        relative_l2=float(torch.linalg.vector_norm(x-y) /
                                          torch.linalg.vector_norm(y).clamp_min(1e-20)))
        for layer in (4, 5, 6, 15):
            operands, stops = fixture(layer)
            initial = [byte_hash(v) for v in operands]
            answer = invoke(layer, operands, stops)
            expected = reference(layer, operands, stops)
            require(answer[4] == expected[4], 'Original fixed join roster')
            metrics = {}
            for index, label in ((0, 'pp'), (2, 'F'), (3, 'core')):
                metrics[label] = err(answer[index], expected[index])
                require(metrics[label]['max_abs'] <= 1e-11 and
                        metrics[label]['relative_l2'] <= 1e-10, 'Direct full-Gram reference: ' + label)
            pp, U, F, core, old = answer
            plane = op.read(torch, pp, operands[2], response, live, U, F, old)
            direct = (pp[response].square() * operands[2]) @ pp[live].T
            metrics['plane'] = err(plane, direct)
            require(metrics['plane']['max_abs'] <= 1e-11 and
                    metrics['plane']['relative_l2'] <= 1e-10, 'Complete signed plane identity')
            require(initial == [byte_hash(v) for v in operands], 'Input values unchanged')
            require(all(bool(torch.isfinite(x).all()) for x in answer[:4] if x is not None),
                    'Every returned tensor finite')
            result['cases'].append(dict(layer=layer, old_columns=old, metrics=metrics,
                                        input_hashes=initial, passed=True))
            print(json.dumps(dict(event='case_pass', layer=layer)), flush=True)
            del operands, answer, expected, pp, U, F, core, plane, direct
            gc.collect()
        operands, stops = fixture(6)
        leaves = tuple(v.clone().requires_grad_(True) for v in operands)
        directions = tuple(torch.sin(torch.arange(v.numel(), dtype=v.dtype).reshape(v.shape) + j + .3)
                           / max(1., v.numel() ** .5) for j, v in enumerate(leaves))
        cot = torch.cos(torch.arange(n * 24, dtype=torch.float64).reshape(n, 24) * .13)
        def scalar(*items):
            pp, U, F, core, old = invoke(6, items, stops)
            plane = op.read(torch, pp, items[2], response, live, U, F, old)
            return (pp*cot).sum() + .07*F.square().sum() + .03*core.square().sum() + .11*plane.square().sum()
        loss = scalar(*leaves)
        gradients = torch.autograd.grad(loss, leaves)
        require(all(bool(torch.isfinite(g).all()) for g in gradients), 'Connected finite complete gradient')
        derivative = sum((g*d).sum() for g, d in zip(gradients, directions))
        epsilon = 1e-5
        with torch.no_grad():
            high = scalar(*(x+epsilon*d for x, d in zip(leaves, directions)))
            low = scalar(*(x-epsilon*d for x, d in zip(leaves, directions)))
            fd = (high-low)/(2*epsilon)
        absolute = float((derivative-fd).abs())
        relative = absolute / max(1e-12, abs(float(derivative)), abs(float(fd)))
        require(absolute <= 1e-8 and relative <= 1e-5, 'Fixed central-difference derivative')
        replay_loss = checkpoint(scalar, *leaves, use_reentrant=False)
        replay_gradients = torch.autograd.grad(replay_loss, leaves)
        require(byte_hash(loss) == byte_hash(replay_loss) and
                all(byte_hash(a) == byte_hash(b) for a, b in zip(gradients, replay_gradients)),
                'Exact nonreentrant checkpoint scalar and complete gradient bytes')
        result['gradient'] = dict(directional=float(derivative), finite_difference=float(fd),
            epsilon=epsilon, absolute_error=absolute, relative_error=relative,
            norms=[float(torch.linalg.vector_norm(g)) for g in gradients],
            checkpoint_scalar_byte_exact=True, checkpoint_gradients_byte_exact=True)
        graph_owners = [weakref.ref(v) for v in leaves] + [weakref.ref(loss), weakref.ref(replay_loss)]
        del operands, leaves, gradients, replay_gradients, replay_loss, loss, high, low, fd, derivative
        del directions, cot, scalar
        gc.collect()
        require(all(ref() is None for ref in graph_owners), 'All gradient input and scalar owners retired')
        require(all(sha(HERE/name) == h for name, h in pins.items()), 'Sources unchanged')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 8_000_000_000, '8GB host ceiling')
        result.update(passed=True, completed=True, source_bindings=pins, hashes_unchanged=True,
                      target_free_operator_only=True)
    except BaseException:
        result['error'] = traceback.format_exc()
    result['elapsed_s'] = time.monotonic()-began
    write(output/'result.json', result)
    rh = sha(output/'result.json')
    write(output/'completion.json', dict(passed=result['passed'], completed=result['completed'], result_sha256=rh))
    write(output/'finalization.json', dict(passed=result['passed'], completed=result['completed'],
        result_sha256=rh, completion_sha256=sha(output/'completion.json'), elapsed_s=time.monotonic()-began))
    signal.alarm(0)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
