"""Approved single target-free current-bank Stage A; NO new consumer GEMMs."""
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
import centered_bank_autodiff_v3 as bound
from tensor_clamp_min_v1 import rewrite_clamps

HERE = Path(__file__).resolve().parent
OUT = HERE/'pair_jet_opportunity_stage_a_v1.json'
CACHE = HERE/'pair_jet_opportunity_stage_a_cache_v1'
TINY = json.loads((HERE/'pair_jet_coalescence_tiny_cost_v2.json').read_text())
assert TINY['passed']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arr(x):
    return x.detach().cpu().numpy().copy()


def construct(p32, q32):
    p, q = p32.astype(np.float64), q32.astype(np.float64)
    d, h = p-q, (p+q)*.5
    d2, h2, hd = (np.sum(d*d, axis=0), np.sum(h*h, axis=0), np.sum(h*d, axis=0))
    gamma = np.divide(hd, d2, out=np.zeros_like(hd), where=d2 > 0)
    r = h-gamma*d
    r2, dr = np.sum(r*r, axis=0), np.sum(d*r, axis=0)
    a = 3*gamma*gamma+.25
    kappa = np.cbrt(a)
    eta = 2*gamma/(kappa*kappa)
    A, B = 1-kappa*eta*eta, eta**3
    # Exact general Gram norm, retaining finite-precision D.R terms.
    ne = A*A*(3*d2*r2*r2+6*dr*dr*r2)-6*A*B*dr*r2*r2+B*B*r2**3
    # Stable centered positive norm, independently certified by root.
    nt = 3*h2*h2*d2+6*h2*hd*hd+1.5*hd*hd*d2+d2**3/16
    ratio2 = np.divide(np.maximum(ne, 0), nt, out=np.zeros_like(ne), where=nt > 0)
    ratio2[(nt == 0) & (ne != 0)] = np.inf
    z32 = (kappa*d+eta*r).astype(np.float32)
    finite = np.all(np.isfinite(z32), axis=0) & np.isfinite(ratio2)
    eligible = (ratio2 <= .01) & finite
    return eligible, z32, dict(gamma=gamma,kappa=kappa,eta=eta,A=A,B=B,
                               tensor_norm_squared=nt,error_norm_squared=ne,
                               relative_tensor_error=np.sqrt(ratio2),
                               transverse_norm_squared=r2,difference_norm_squared=d2,
                               orthogonality_inner_product=dr)


prefix_source = bound.parent.prefix_source
marker = '    host, C, target, new_radial, aux = nonlinear(mu, Cpre, k21, radial)'
assert prefix_source.count(marker) == 1
prefix_source = prefix_source.replace(marker,
    '    self.capture_pre(l, mu, Cpre, k21, radial, hstate, alive)\n'+marker)
prefix_code, clamps = rewrite_clamps(prefix_source, __file__+':capture_prefix')
assert clamps == 2
prefix_env = dict(bound._prefix_env)
exec(prefix_code, prefix_env)


class CaptureCard(bound.ExplicitBoundBankFullCard):
    def __init__(self, rank):
        super().__init__(rank, 'mps', audit_selections=True)
        self.saved = {}
        self.rows = []
        self.counts = []
        self.sites = {x['decision_post_layer']: x for x in
                      next(r for r in TINY['cost']['rows'] if r['rank']==rank)['cohorts']
                      if x['statically_profitable']}

    def step(self, l, state, control, P):
        fn = prefix_env['step'] if l < self.depth-1 else bound._final_env['step']
        return fn(self,l,state,control,P)

    def capture_pre(self, l, mu, Cpre, k21, radial, hstate, alive):
        birth = l-2
        if birth in self.saved:
            cache = self.saved[birth]
            for key, value in dict(next_mu=mu,next_Cpre=Cpre,next_K21=k21,
                                   next_radial_input=radial,next_hidden_input=hstate,
                                   next_alive=alive,next_w=self.w[l],next_fixed=self.fixed[l]).items():
                cache[key] = arr(value)

    def capture_post(self, l, state):
        prior = l-2
        if prior in self.saved:
            for key, value in dict(next_post_mean=state[0],next_post_covariance=state[1],
                                   next_post_radial=state[7]).items():
                self.saved[prior][key] = arr(value)
        if l not in self.sites:
            return
        site = self.sites[l]
        birth, count = l-1, self.counts[l-1]
        assert site['pairs'] == count
        offset = 2*sum(self.counts[:birth])
        p32 = arr(state[2][:,offset:offset+count])
        q32 = arr(state[2][:,offset+count:offset+2*count])
        lam = arr(state[3][offset:offset+count])
        other_lam = arr(state[3][offset+count:offset+2*count])
        assert np.array_equal(lam,-other_lam)
        assert p32.shape == q32.shape == (1024,count)
        eligible,z32,details = construct(p32,q32)
        ix = np.flatnonzero(eligible)
        row = dict(birth_layer=birth,decision_post_layer=l,original_pairs=count,
                   eligible_pairs=len(ix),fraction_eligible=float(eligible.mean()),
                   source_offset=offset,weight_pairing_equal=True,
                   relative_tensor_error_quantiles=np.quantile(details['relative_tensor_error'],[0,.25,.5,.75,1]).tolist(),
                   negative_error_norms=int(np.count_nonzero(details['error_norm_squared'] < 0)),
                   source_active=int(state[6].sum().item()))
        self.rows.append(row)
        if len(ix):
            self.saved[birth] = dict(eligible_indices=ix,original_plus=p32[:,ix],
                                     original_minus=q32[:,ix],merged_z=z32[:,ix],
                                     weight=lam[ix],source_alive=arr(state[6]))
            for key,value in details.items():
                self.saved[birth][key] = value[ix]
        print(json.dumps(dict(stage='eligibility',**row)),flush=True)


def price(rank, counts, active, rows):
    coef = json.loads((HERE/'direct_kernel_schedule_prices_v1.json').read_text())['coefficients']
    measured = next(x['flops'] for x in json.loads((HERE/'fullpair_runtime_rectd4_bulkcopies_openblas_series_v1.json').read_text())['records'] if x['rank']==rank)

    def mm(m,k,n):
        if not m or not k or not n: return 0.
        assert all(t % 32 == 0 for t in (m,k,n))
        if n > 8192:
            return sum(mm(m,k,min(8192,n-j)) for j in range(0,n,8192))+m*n
        if k > 8192:
            return sum(mm(m,min(8192,k-j),n) for j in range(0,k,8192))+(len(range(0,k,8192))-1)*m*n
        depths = (3,4,5) if max(m,k,n)<=1024 else (4,)
        return min(2*m*k*n-m*n,*[sum(a*b for a,b in zip(coef[str(d)],(m*k*n,m*k,k*n,m*n))) for d in depths])

    def products(widths):
        saved=[]
        for l in range(1,16):
            old = sum(widths[:l-1]); new = counts[l-1]
            push=mm(1024,int(active[l-1]),old)+mm(1024,int(active[l-1]),new)
            plane=mm(int(active[l]),old+2*new,int(active[l])) if l<15 else 0.
            saved.append(dict(layer=l,inherited_columns=old,push_flops=push,plane_flops=plane))
        return sum(x['push_flops']+x['plane_flops'] for x in saved),saved

    original_widths=[2*n for n in counts]
    widths=original_widths.copy()
    restoration=construction=copies=0.
    for row in rows:
        b, n, e, a = row['birth_layer'],row['original_pairs'],row['eligible_pairs'],row['source_active']
        widths[b] = ((2*n-e+31)//32)*32
        # Pad each whole compressed cohort to32, with zero-weight atoms.
        # This preserves the existing kernels rather than falling back to
        # their nonmultiple32 plain-product branch.
        epad=((e+31)//32)*32
        restore=mm(a,2*epad,a)
        build=256*a*n+512*n+(8*a*a if e else 0)
        extra_copy=(4*1024*(2*n+widths[b])+8*a*epad) if e else 0
        row.update(padded_cohort_columns=widths[b],padded_restoration_pairs=epad,
                   restoration_gemm_projection=restore,construction_all_tested_pairs=build,
                   extra_packing_copy_allowance=extra_copy)
        restoration+=restore;construction+=build;copies+=extra_copy
    old,old_layers=products(original_widths)
    new,new_layers=products(widths)
    reference=next(r['original_cp_gemm_projection'] for r in TINY['cost']['rows'] if r['rank']==rank)
    assert old == reference, (rank,old,reference)
    saving=old-new-restoration-construction-copies
    return dict(passed=bool(saving/measured>=.25),whole_bank_saving_fraction=saving/measured,
                net_saving_projection=saving,parent_measured_first_context_flops=measured,
                parent_cp_gemm_projection=old,proposed_cp_gemm_projection=new,
                restoration_gemm_projection=restoration,construction_projection=construction,
                extra_packing_copy_allowance=copies,original_layers=old_layers,proposed_layers=new_layers,
                padded_cohort_widths=widths)


def main():
    assert not OUT.exists() and not CACHE.exists()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','0') == '0'
    assert 'threadpoolctl' not in sys.modules
    torch.set_num_threads(3)
    assert torch.backends.mps.is_available()
    bank_path=HERE/'candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py'
    assert sha(bank_path)=='3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757'
    start=time.time();CACHE.mkdir()
    records=[];max_gpu=0
    for rank in (35,100):
        card=CaptureCard(rank)
        with torch.no_grad():
            state=card.initial();zero=torch.zeros(1024,device='mps');means=[]
            for l in range(16):
                state=card.step(l,state,zero,card.P0)
                if l<15:
                    card.counts.append(len(card.selections[2*l+1][0]))
                card.capture_post(l,state)
                means.append(np.maximum(arr(state[0]),0))
                max_gpu=max(max_gpu,int(torch.mps.driver_allocated_memory()))
        baseline=np.load(HERE/'centered_parent_numerics_v1.npz')[f'baseline_{rank}']
        maximum=float(np.max(np.abs(np.stack(means)-baseline)))
        maskfile=np.load(HERE/f'full_carrier_local_cell_r{rank}_v1.npz')['baseline_masks']
        actual_masks=np.stack([x[1] for x in card.selections])
        equal=bool(np.array_equal(actual_masks,maskfile[:30]))
        counts_ref=[x['count'] for x in next(r for r in json.loads((HERE/'cohort_fixed_count_thinning_wide_v1.json').read_text())['records'] if r['rank']==rank)['cohorts']]
        assert maximum<=2e-6 and equal and card.counts==counts_ref
        active=maskfile[::2].sum(1).astype(int)
        projected=price(rank,card.counts,active,card.rows)
        files=[];cache_bytes=0
        for birth,cache in card.saved.items():
            assert 'next_post_mean' in cache and 'next_K21' in cache
            path=CACHE/f'r{rank}_birth{birth:02d}.npz'
            assert not path.exists()
            np.savez(path,**cache)
            cache_bytes+=sum(x.nbytes for x in cache.values())
            files.append(dict(name=path.name,sha256=sha(path),bytes=path.stat().st_size))
        parity_path=CACHE/f'r{rank}_baseline.npz'
        np.savez(parity_path,means=np.stack(means),masks=actual_masks,birth_counts=np.array(card.counts))
        row=dict(rank=rank,baseline_max_abs=maximum,memberships_equal=equal,
                 birth_counts=card.counts,eligible_total=sum(x['eligible_pairs'] for x in card.rows),
                 tested_total=sum(x['original_pairs'] for x in card.rows),sites=card.rows,
                 projection=projected,cache_raw_bytes=cache_bytes,cache_files=files,
                 baseline_cache_sha256=sha(parity_path))
        records.append(row)
        print(json.dumps(dict(stage='rank_complete',rank=rank,baseline_max_abs=maximum,
                              memberships_equal=equal,eligible_total=row['eligible_total'],
                              tested_total=row['tested_total'],cache_raw_bytes=cache_bytes,
                              whole_bank_saving_fraction=projected['whole_bank_saving_fraction'],
                              passed=projected['passed'])),flush=True)
        del card,state,baseline,actual_masks
        gc.collect();torch.mps.empty_cache()
    source_names=['candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py',
                  'payload_compact_feed_fullpair_v1.npz','payload_fullpair_final_head_v1.npz',
                  'centered_bank_autodiff_v3.py','centered_bank_autodiff_v1.py',
                  'centered_torch_numerics_v3.py','certify_pair_jet_coalescence_v1.py',
                  'pair_jet_coalescence_tiny_cost_v2.json',Path(__file__).name]
    result=dict(passed=all(r['projection']['passed'] for r in records),records=records,
                target_access=False,stage_b_executed=False,additional_consumer_gemms=False,
                source_sha256={name:sha(HERE/name) for name in source_names},
                utc_finished=datetime.now(timezone.utc).isoformat(),wall_seconds=time.time()-start,
                peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                largest_observed_gpu_driver_bytes=max_gpu,
                qualification='ONE coherent current-bank MPS35/100 capture with fixed first-transport .1 ideal tensor criterion and finite actualf32Z. All30 memberships and baseline means checked. Padding,2Nrestoration,all-tested construction and extra packing charged analytically; no measured new candidate bill/runtime. NO extra consumer GEMMs, target data, modified rollout or StageB. CPU/process and GPU peaks are nonadditive.')
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    main()
