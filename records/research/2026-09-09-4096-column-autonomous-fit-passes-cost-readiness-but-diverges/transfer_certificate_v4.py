"""Fixed CPU/MPS host-transfer and dropped-plane gradient prerequisite; inert.

No network, payload, truth or fitting. Reuse the original dyadic width8 data,
and exercise the repaired model's complete retirement once on each device.
"""
import argparse
from datetime import datetime, timezone
import os
import resource
import signal
import sys
import time
import traceback
from common import HERE, bindings, completed, module, sha, unchanged, write


def check(np, torch, model):
    tiny = module(HERE/'check_tiny.py', 'cap4096_fixed_transfer_fixture')
    p, lam, nb, pos, neg, live = tiny.fixture(np)
    first, second = nb[2].copy(), nb[2].copy()
    first[nb[1], np.arange(3)] = nb[3]
    second[nb[1], np.arange(3)] = nb[4]
    physical = np.concatenate((p, first, second), axis=1)
    literal_plane = np.einsum('ir,ir,jr,r->ij', physical.astype(float), physical.astype(float),
                              physical.astype(float), lam.astype(float))
    allowed = np.zeros((8, 8), bool)
    allowed[np.ix_(live, live)] = True
    np.fill_diagonal(allowed, True)
    cp = np.where(allowed, literal_plane, 0)
    cot = ((np.arange(64).reshape(8, 8)%13)-6).astype(np.float32)/32
    records, saved = [], {}
    for device in ('cpu', 'mps'):
        x = torch.tensor(physical, device=device, dtype=torch.float32, requires_grad=True)
        weights = torch.tensor(lam, device=device, dtype=torch.float32, requires_grad=True)
        index = lambda z: torch.tensor(z, device=device, dtype=torch.long)
        # Exact promotion of the stored F32 values, detached only for ranking.
        for name, value, expected in (('positive', x.detach()[:, index(pos)], physical[:, pos]),
                                     ('negative', x.detach()[:, index(neg)], physical[:, neg]),
                                     ('lambda', weights.detach(), lam)):
            converted = value.cpu().to(torch.float64)
            assert not converted.requires_grad
            assert converted.numpy().tobytes() == expected.astype(np.float64).tobytes(), name
        dtype = torch.float64 if device == 'cpu' else torch.float32
        answer = model.retire(x, weights, index(pos), index(neg), 3,
            torch.tensor(cp, device=device, dtype=dtype), index(live), 4)
        assert answer[5]['chosen'] == [0, 1, 3, 4]
        assert answer[5]['minimum_relative_margin'] > 0
        # Dyadic functional exercises both survivors and ALL dropped-plane
        # gradient paths, including the explicit CPU64 diagonal return to MPS.
        loss = (answer[4]*torch.tensor(cot, device=device, dtype=dtype)).sum()
        loss = loss+(answer[0]*answer[0]*answer[1]).sum()/64
        loss.backward()
        gx, gw = x.grad.detach().cpu().numpy().copy(), weights.grad.detach().cpu().numpy().copy()
        value = answer[4].detach().cpu().numpy().astype(float)
        assert np.isfinite(gx).all() and np.isfinite(gw).all() and np.isfinite(value).all()
        assert np.linalg.norm(gx[:, [2, 5, 8, 11]].astype(float)) > 1e-10
        saved[device] = (value, gx.astype(float), gw.astype(float))
        records.append(dict(device=device, detached_promotions_byte_exact=True,
            chosen=answer[5]['chosen'], gradient_norm=float(np.linalg.norm(gx.astype(float))),
            dropped_gradient_norm=float(np.linalg.norm(gx[:, [2, 5, 8, 11]].astype(float))),
            mps_driver_bytes=int(torch.mps.driver_allocated_memory())))
        del x, weights, answer, loss
    reference, observed = saved['cpu'], saved['mps']
    value_error = float(np.max(np.abs(reference[0]-observed[0])))
    cpu_gradient = np.concatenate((reference[1].ravel(), reference[2].ravel()))
    mps_gradient = np.concatenate((observed[1].ravel(), observed[2].ravel()))
    gradient_error = float(np.max(np.abs(cpu_gradient-mps_gradient)))
    gradient_relative = float(np.linalg.norm(cpu_gradient-mps_gradient)/np.linalg.norm(cpu_gradient))
    assert value_error <= 1e-12
    assert gradient_error <= 1e-6 and gradient_relative <= 1e-6
    return dict(devices=records, plane_max_error=value_error,
        gradient_max_error=gradient_error, gradient_relative=gradient_relative,
        detached_ranking_and_differentiable_diagonal_exercised=True,
        no_network_or_payload=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--identity-sha256')
    args = parser.parse_args()
    if not args.run:
        parser.print_help(); return 0
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    assert 'threadpoolctl' not in sys.modules
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0'
    out = HERE/'transfer_v4'
    assert not out.exists()
    identity = completed(HERE/'identity_v3', args.identity_sha256)
    assert identity['schema'] == 'cap4096_identity_certificate_v3'
    pins = bindings((__file__, HERE/'common.py', HERE/'model.py', HERE/'model_v2.py',
        HERE/'check_tiny.py', HERE/'PROTOCOL.md', HERE/'TRANSFER_REPAIR_V4.md',
        HERE/'identity_v3/result.json', HERE/'identity_v3/completion.json'))
    out.mkdir()
    start = time.monotonic()
    result = dict(passed=False, schema='cap4096_transfer_certificate_v4', source_bindings=pins,
        identity_result_sha256=args.identity_sha256, network_accessed=False, truth_accessed=False,
        hard_wall_limit_s=60, rss_limit_bytes=2000000000, mps_driver_limit_bytes=2000000000,
        started_utc=datetime.now(timezone.utc).isoformat())
    write(out/'started.json', result)
    torch_module = None
    def watchdog(signum, frame):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        driver = (torch_module.mps.driver_allocated_memory() if torch_module is not None else 0)
        if time.monotonic()-start >= 60 or rss >= 2000000000 or driver >= 2000000000:
            signal.setitimer(signal.ITIMER_REAL, 0)
            raise RuntimeError('Transfer certificate60s/2GB resource limit')
    previous = signal.signal(signal.SIGALRM, watchdog)
    signal.setitimer(signal.ITIMER_REAL, .1, .1)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        assert torch.backends.mps.is_available()
        torch_module = torch
        import model_v2 as model
        result['checks'] = check(np, torch, model)
        torch.mps.synchronize()
        result['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result['mps_driver_bytes'] = int(torch.mps.driver_allocated_memory())
        assert result['peak_rss_bytes'] < 2000000000 and result['mps_driver_bytes'] < 2000000000
        assert time.monotonic()-start < 60
        unchanged(pins)
        result['passed'] = True
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    result['wall_s'] = time.monotonic()-start
    write(out/'result.json', result)
    write(out/'completion.json', dict(passed=result['passed'], result_sha256=sha(out/'result.json')))
    print(result, flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
