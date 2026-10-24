"""Tiny signed-pair tensor identity and one fixed all-eligible cost envelope."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'pair_jet_coalescence_tiny_cost_v1.json'


def cube(p):
    return np.einsum('i,j,k->ijk', p, p, p)


def sym(d, r):
    return (np.einsum('i,j,k->ijk', d, r, r)
            + np.einsum('i,j,k->ijk', r, d, r)
            + np.einsum('i,j,k->ijk', r, r, d))


def gram(x, y, z):
    return np.array([
        [x**3, 3*x*x*z, 3*x*z*z, z**3],
        [3*x*x*z, 3*x*x*y+6*x*z*z, 6*x*y*z+3*z**3, 3*y*z*z],
        [3*x*z*z, 6*x*y*z+3*z**3, 3*x*y*y+6*y*z*z, 3*y*y*z],
        [z**3, 3*y*z*z, 3*y*y*z, y**3]])


def coalesce(p, q):
    d, h = p-q, (p+q)*.5
    x = float(d@d)
    gamma = float(h@d)/x if x else 0.
    r = h-gamma*d
    a = 3*gamma*gamma+.25
    kappa = np.cbrt(a)
    eta = 2*gamma/(kappa*kappa)
    A, B = 1-kappa*eta*eta, eta**3
    merged = kappa*d+eta*r
    G = gram(x, float(r@r), float(d@r))
    tc, ec = np.array([a, 2*gamma, 1., 0.]), np.array([0., 0., A, -B])
    norm_t, norm_e = float(tc@G@tc), float(ec@G@ec)
    return merged, d, r, A, B, norm_t, norm_e


def star_tensor(plane):
    n = len(plane)
    t = np.zeros((n, n, n))
    for i in range(n):
        t[i, i, i] = plane[i, i]
        for j in range(n):
            if i != j:
                t[i, i, j] = t[i, j, i] = t[j, i, i] = plane[i, j]
    return t


def star_push(plane, w):
    star = 3*plane.T-2*np.diag(np.diag(plane))
    u = w.T
    ua = u@star
    return (2*(ua*u)@u.T+(u*u)@ua.T)/3


def tiny():
    rng = np.random.Generator(np.random.PCG64DXSM(930506))
    n = 7
    fixtures = []
    for gamma, transverse, length in [(-8., .3, 1.), (-1., 1., 1.),
                                     (0., .2, 1.), (.5, .4, 1.),
                                     (1., .1, 1.), (8., .4, 1.),
                                     (2., 0., 1.), (2., .1, 1e-5)]:
        d = rng.normal(size=n); d *= length/np.linalg.norm(d)
        r = rng.normal(size=n); r -= d*(d@r)/(d@d)
        r *= transverse/max(np.linalg.norm(r), 1e-30)
        h = gamma*d+r
        fixtures.append((h+.5*d, h-.5*d))
    same = rng.normal(size=n)
    fixtures.append((same, same.copy()))
    records = []
    for j, (p, q) in enumerate(fixtures):
        weight = (-1.)**j*(.4+.1*j)
        z, d, r, A, B, nt, ne = coalesce(p, q)
        # Stable polynomial T is the primary reference for close pairs.
        h = (p+q)*.5
        tensor = weight*(sym(d, h)+.25*cube(d))
        error = weight*(A*sym(d, r)-B*cube(r))
        delta = tensor-weight*cube(z)
        scale = max(np.linalg.norm(tensor), 1e-30)
        plane = weight*(np.outer(r*r, A*d-B*r)+np.outer(d*r, 2*A*r))
        dense_plane = np.einsum('iij->ij', error)
        w = rng.normal(size=(n, n))/np.sqrt(n)
        push = np.einsum('ijk,ia,jb,kc->abc', error, w, w, w)
        dz, rz = w.T@d, w.T@r
        next_plane = weight*(np.outer(rz*rz, A*dz-B*rz)+np.outer(dz*rz, 2*A*rz))
        restored = star_tensor(plane)
        next_residual = np.einsum('ijk,ia,jb,kc->abc', error-restored, w, w, w)
        scale_next = max(np.linalg.norm(next_residual), np.linalg.norm(push), 1e-30)
        row = dict(fixture=j,
                   tensor_identity_error=float(np.linalg.norm(delta-error)/scale),
                   tensor_norm_error=float(abs(weight*weight*nt-np.sum(tensor*tensor))/max(scale*scale, 1e-60)),
                   error_norm_error=float(abs(weight*weight*ne-np.sum(error*error))/max(scale*scale, 1e-60)),
                   repeated_plane_error=float(np.linalg.norm(plane-dense_plane)/scale),
                   next_plane_error=float(np.linalg.norm(next_plane-np.einsum('iij->ij', push))/max(np.linalg.norm(push), 1e-30)),
                   restored_next_plane_error=float(np.linalg.norm(next_plane-star_push(plane,w)-np.einsum('iij->ij', next_residual))/scale_next),
                   relative_tensor_error=float(np.sqrt(max(ne, 0)/nt)) if nt > 0 else 0.)
        row['passed'] = all(v <= 1e-11 for k, v in row.items() if k.endswith('_error') and k != 'relative_tensor_error')
        records.append(row)
    return dict(passed=all(r['passed'] for r in records), records=records)


def cost():
    coef = json.loads((HERE/'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
    bank = json.loads((HERE/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records']
    coherent = json.loads((HERE/'cohort_fixed_count_thinning_wide_v1.json').read_text())['records']
    prior = json.loads((HERE/'uniform_half_cp_shape_cost_v1.json').read_text())['records']

    def mm(m, k, n):
        if not m or not k or not n:
            return 0.
        if n > 8192:
            return sum(mm(m, k, min(8192, n-j)) for j in range(0, n, 8192))+m*n
        if k > 8192:
            return sum(mm(m, min(8192, k-j), n) for j in range(0, k, 8192))+(len(range(0,k,8192))-1)*m*n
        plain = 2*m*k*n-m*n
        if any(t % 32 for t in (m,k,n)):
            return float(plain)
        depths = (3,4,5) if max(m,k,n) <= 1024 else (4,)
        prices = [sum(a*b for a,b in zip(coef[str(d)],(m*k*n,m*k,k*n,m*n))) for d in depths]
        return min(plain,*prices)

    rows = []
    for rank in (35, 100):
        counts = [r['count'] for r in next(x for x in coherent if x['rank']==rank)['cohorts']]
        masks = np.load(HERE/f'full_carrier_local_cell_r{rank}_v1.npz')['baseline_masks']
        active = masks[::2].sum(1).astype(int)
        assert len(active) == 16 and sum(counts) == next(x for x in coherent if x['rank']==rank)['pair_count']
        whole = next(x['flops'] for x in bank if x['rank']==rank)

        def products(merged):
            records = []
            for l in range(1, 16):
                inherited = sum(counts[b]*(1 if merged[b] else 2) for b in range(l-1))
                new = counts[l-1]
                push = mm(1024, int(active[l-1]), inherited)+mm(1024,int(active[l-1]),new)
                read = mm(int(active[l]), inherited+2*new, int(active[l])) if l < 15 else 0.
                records.append(dict(layer=l, inherited_columns=inherited, push_flops=push, plane_flops=read))
            return sum(r['push_flops']+r['plane_flops'] for r in records), records

        parent, parent_layers = products([False]*15)
        expected = next(x['original_cp_gemm_projection'] for x in prior if x['rank']==rank)
        assert parent == expected, (rank,parent,expected)
        selected = [False]*15
        cohorts = []
        # One predetermined eligibility site. Cost-positive is a static
        # admissibility rule, not an age/rank parameter or a search sweep.
        for b in range(14):
            a, count = int(active[b+1]), counts[b]
            restoration = mm(a, 2*count, a)
            overhead = 256*a*count+512*count+8*a*a
            probe = [False]*15; probe[b] = True
            alone, _ = products(probe)
            gross = parent-alone
            selected[b] = bool(gross > restoration+overhead)
            cohorts.append(dict(birth_layer=b, decision_post_layer=b+1, pairs=count,
                                isolated_future_gemm_saving=gross, restoration_gemm=restoration,
                                construction_and_assembly_allowance=overhead,
                                statically_profitable=selected[b]))
        reduced, layers = products(selected)
        restoration = sum(r['restoration_gemm'] for r in cohorts if r['statically_profitable'])
        overhead = sum(r['construction_and_assembly_allowance'] for r in cohorts if r['statically_profitable'])
        net = parent-reduced-restoration-overhead
        rows.append(dict(rank=rank, active=active.tolist(), birth_counts=counts,
                         original_cp_gemm_projection=parent, reduced_cp_gemm_projection=reduced,
                         restoration_gemm_projection=restoration,
                         acquisition_and_assembly_allowance=overhead,
                         net_saving_projection=net, measured_bank_first_context_flops=whole,
                         whole_bank_saving_fraction=net/whole, passed=bool(net/whole >= .25),
                         cohorts=cohorts, original_layers=parent_layers, reduced_layers=layers))
    return dict(passed=all(r['passed'] for r in rows), rows=rows,
                qualification='ONE all-eligible analytical envelope with statically profitable merges only, first-transport decision, exact parent projection reproduced. Actual tensor eligibility, padded partial counts, nonlinear feedback, real FNP bill/runtime and raw unknown. No bank work executed.')


def main():
    assert not OUT.exists()
    start = time.time()
    certificate = tiny()
    assert certificate['passed'], certificate
    projection = cost()
    result = dict(passed=certificate['passed'] and projection['passed'], tiny=certificate,
                  cost=projection, target_access=False, wide_execution=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.time()-start,
                  peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'tiny':{'passed':certificate['passed']},
                     'cost':{**projection,'rows':[{k:v for k,v in r.items() if k not in ('cohorts','original_layers','reduced_layers')} for r in projection['rows']]}},indent=2))


if __name__ == '__main__':
    main()
