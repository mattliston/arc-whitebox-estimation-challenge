"""One explicitly authorized frozen A or B energy-information stream.

AP/Torch/MPS acquisition only. Never imports threadpoolctl or a card.
No sample, seed, layer, bin, batch, or threshold override is accepted.
Default invocation cannot execute; root must supply the review flag.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

from cross_depth_energy_statistics_v1 import (
    HERE, WIDTH, SAMPLES, BATCH, BLOCKS, BLOCK_SIZE, RANK, SEED, XORS,
    RNG_SEEDS, THRESHOLDS, WEIGHT_ROW, SHIFT_INPUT, PRIOR_HASHES,
    BlockSums, file_record, sha256, utc_now, verify_frozen_inputs,
    norm_sums, moments, scalar_empirical_algebra, legacy_finish,
    statistic_certificate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', choices=('A','B'), required=True)
    parser.add_argument('--execute-reviewed-protocol', action='store_true')
    args = parser.parse_args()
    if not args.execute_reviewed_protocol:
        parser.error('Acquisition requires root review and explicit --execute-reviewed-protocol.')
    if 'threadpoolctl' in sys.modules:
        raise RuntimeError('AP must never import threadpoolctl')
    nice = os.getpriority(os.PRIO_PROCESS,0)
    if nice < 10:
        raise RuntimeError(f'launch this acquisition at nice >=10; observed {nice}')

    out = HERE/f'cross_depth_energy_r35_{args.stream}_v1.npz'
    receipt_path = out.with_suffix('.json')
    log_path = out.with_suffix('.progress.jsonl')
    failure_path = out.with_suffix('.failure.json')
    for path in (out,receipt_path,log_path,failure_path):
        if path.exists():
            raise FileExistsError(f'preserve existing artifact; refuse overwrite: {path}')
    started = time.time()
    target_access_started = False

    def record(item):
        row = dict(utc=utc_now(),elapsed_seconds=time.time()-started,**item)
        with log_path.open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)

    try:
        # Check metadata/bank inputs and actual accumulator algebra before
        # importing Torch, opening reference target arrays, or drawing samples.
        provenance = verify_frozen_inputs()
        small = statistic_certificate()
        for path in (Path(__file__),HERE/'cross_depth_energy_statistics_v1.py',
                     HERE/'judge_cross_depth_energy_v1.py'):
            provenance[str(path)] = file_record(path)
        prior_path = HERE/f'residual_one_high_truth_r35_l12_{args.stream}_v1.npz'
        if sha256(prior_path) != PRIOR_HASHES[args.stream]:
            raise ValueError('published prior stream digest mismatch')
        provenance[str(prior_path)] = file_record(prior_path)
        target_access_started = True
        with np.load(prior_path,allow_pickle=False) as f:
            prior = {key:f[key].copy() for key in
                     ('P','batch','seed','seed_xor','indices','post_shift_actual',
                      'next_shift_actual','post_mean_residual','next_pre_mean_residual','next_K21')}
        xor = XORS[args.stream]
        if (int(prior['P']) != SAMPLES or int(prior['batch']) != BATCH
                or int(prior['seed']) != SEED or int(prior['seed_xor']) != xor
                or ((SEED^xor)&((1<<64)-1)) != RNG_SEEDS[args.stream]
                or not np.array_equal(prior['indices'],np.arange(0,WIDTH,32))):
            raise ValueError('prior count, seed, batch, or index law differs from protocol')
        # The historical oracle input is ONLY a source of replay shifts.
        # Its teacher-forced moments are never used as a comparator.
        with np.load(SHIFT_INPUT,allow_pickle=False) as f:
            shift_post = f['post_shift'].astype(np.float32)
            shift_next = f['next_pre_shift'].astype(np.float32)
        if (not np.array_equal(shift_post,prior['post_shift_actual'])
                or not np.array_equal(shift_next,prior['next_shift_actual'])):
            raise ValueError('legacy actual shift rounding differs')

        import torch
        if 'threadpoolctl' in sys.modules:
            raise RuntimeError('forbidden threadpoolctl import in AP')
        if not torch.backends.mps.is_available():
            raise RuntimeError('frozen stream requires the original MPS backend')
        torch.set_num_threads(3)
        dev = torch.device('mps')
        weights = np.load(WEIGHT_ROW/'weights.npy',mmap_mode='r',allow_pickle=False)
        if weights.shape != (16,WIDTH,WIDTH) or weights.dtype != np.dtype('float32'):
            raise ValueError('wrong authenticated weight shape/dtype')
        # Keep the original from_numpy -> MPS weight construction.
        wg = [torch.from_numpy(np.array(weights[l],np.float32,copy=True)).to(dev)
              for l in range(14)]
        sg = torch.tensor(shift_post,dtype=torch.float32,device=dev)
        yg = torch.tensor(shift_next,dtype=torch.float32,device=dev)
        indices = prior['indices'].copy()
        ig = torch.tensor(indices,dtype=torch.long,device=dev)
        stats = BlockSums(shift_next)
        post_sum = np.zeros(WIDTH)
        next_sum = np.zeros(WIDTH)
        next_second = np.zeros((len(indices),WIDTH))
        next_square = np.zeros(WIDTH)
        next_third = np.zeros_like(next_second)
        norm_mismatch = np.zeros(3,np.int64)  # A4, A8, joint four-bin membership.
        done = 0
        batches = 0
        rng = np.random.Generator(np.random.PCG64DXSM((SEED^xor)&((1<<64)-1)))
        record(dict(event='qualified_before_sampling',stream=args.stream,rank=RANK,
                    samples=SAMPLES,batch=BATCH,nice=nice,torch_threads=torch.get_num_threads(),
                    torch_version=torch.__version__,statistics_certificate=small,
                    thresholds=THRESHOLDS.tolist(),rng_seed=RNG_SEEDS[args.stream]))
        with torch.no_grad():
            while done < SAMPLES:
                count = min(BATCH,SAMPLES-done)
                # Exactly the original RNG call, with no block-boundary splits.
                cpu_input = rng.standard_normal((count,WIDTH),dtype=np.float32)
                norm64,norm32 = norm_sums(cpu_input)
                if (not np.all(np.isfinite(norm64)) or np.any(norm64 <= 0)
                        or not np.all(np.isfinite(norm32)) or np.any(norm32 <= 0)):
                    raise ValueError('invalid input norm; no repair permitted')
                x = torch.from_numpy(cpu_input).to(dev)
                del cpu_input
                bit4 = bit4_alt = bit8 = bit8_alt = None
                for layer,w in enumerate(wg[:-1]):
                    # Do not replace this with the repaired card's distinct
                    # clamp implementation: it replays the historical sampler.
                    x = torch.clamp(x@w,min=0)
                    if layer in (4,8):
                        cpu_hidden = x.cpu().numpy()
                        hn64,hn32 = norm_sums(cpu_hidden)
                        del cpu_hidden
                        energy64 = WIDTH*hn64/norm64
                        energy32 = WIDTH*hn32/norm32
                        if not np.all(np.isfinite(energy64)):
                            raise ValueError('invalid hidden energy')
                        bit = energy64 > THRESHOLDS[0 if layer == 4 else 1]
                        alternate = energy32 > THRESHOLDS[0 if layer == 4 else 1]
                        if layer == 4:
                            bit4,bit4_alt = bit,alternate
                        else:
                            bit8,bit8_alt = bit,alternate
                        del hn64,hn32,energy64,energy32
                memberships = 2*bit4.astype(np.int8)+bit8.astype(np.int8)
                memberships_alt = 2*bit4_alt.astype(np.int8)+bit8_alt.astype(np.int8)
                norm_mismatch += np.array([np.count_nonzero(bit4 != bit4_alt),
                                           np.count_nonzero(bit8 != bit8_alt),
                                           np.count_nonzero(memberships != memberships_alt)])
                del bit4,bit4_alt,bit8,bit8_alt,memberships_alt,norm64,norm32

                # Original post12 / pre13 replay reductions and products.
                h = x-sg
                post_sum += h.sum(0).cpu().numpy().astype(np.float64)
                del h
                z = x@wg[-1]
                del x
                y = z-yg
                yi = y.index_select(1,ig)
                next_sum += y.sum(0).cpu().numpy().astype(np.float64)
                next_second += (yi.T@y).cpu().numpy().astype(np.float64)
                next_square += (y*y).sum(0).cpu().numpy().astype(np.float64)
                next_third += ((yi*yi).T@y).cpu().numpy().astype(np.float64)
                del y,yi

                # New statistics use actual Z, not a float32 undo of the shift.
                cpu_z = z.cpu().numpy()
                del z
                stats.update(cpu_z,memberships)
                del cpu_z,memberships
                done += count
                batches += 1
                torch.mps.synchronize()
                torch.mps.empty_cache()
                peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                if sys.platform != 'darwin':
                    peak *= 1024
                if peak > 2*(1<<30):
                    raise MemoryError(f'fixed resource qualification exceeded 2 GiB RSS: {peak}')
                if batches % 8 == 0 or done == SAMPLES:
                    record(dict(event='sampling',stream=args.stream,samples=done,batches=batches,
                                peak_rss_bytes=peak,
                                mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                                mps_driver_allocated_bytes=torch.mps.driver_allocated_memory()))

        if (stats.samples != SAMPLES
                or not np.array_equal(np.sum(stats.counts,axis=1),np.full(BLOCKS,BLOCK_SIZE))):
            raise ValueError('fixed count/block partition violated')
        legacy = legacy_finish(post_sum,next_sum,next_second,next_square,next_third,indices,SAMPLES)
        legacy_parity = {key:bool(np.array_equal(value,prior[key])) for key,value in legacy.items()}
        legacy_max = {key:float(np.max(np.abs(value-prior[key]))) for key,value in legacy.items()}
        total = moments(SAMPLES,stats.global_sums,stats.shift)
        reference_diag = prior['next_K21'][np.arange(len(indices)),indices]
        difference = total['third'][indices]-reference_diag
        relative_third = float(np.linalg.norm(difference)/np.linalg.norm(reference_diag))
        algebra = scalar_empirical_algebra(stats.counts,stats.sums,stats.global_sums,stats.shift)
        bin_counts = np.sum(stats.counts,axis=0)
        conditional = moments(bin_counts,np.sum(stats.sums,axis=0),stats.shift)
        qualifications = dict(legacy_byte_parity=all(legacy_parity.values()),
                              cpu_legacy_third=relative_third <= 1e-4,
                              empirical_algebra=max(algebra.values()) <= 1e-11,
                              bin_support=bool(np.all(bin_counts >= 8192)),
                              finite_statistics=bool(np.all(np.isfinite(stats.sums))
                                                     and np.all(np.isfinite(stats.global_sums))),
                              nonnegative_conditional_variance=bool(np.all(conditional['variance'] >= 0)),
                              exact_block_counts=True,resource_limit=True)
        passed = all(qualifications.values())
        for path,item in provenance.items():
            if sha256(path) != item['sha256']:
                raise ValueError(f'source/input changed during acquisition: {path}')
        arrays = dict(block_counts=stats.counts,block_sums=stats.sums,global_sums=stats.global_sums,
                      shift=stats.shift,post_shift_actual=shift_post,next_shift_actual=shift_next,
                      thresholds=THRESHOLDS,indices=indices,norm_membership_mismatch=norm_mismatch,
                      legacy_post_sum=post_sum,legacy_next_sum=next_sum,
                      legacy_next_second=next_second,legacy_next_square=next_square,
                      legacy_next_third=next_third,**legacy,
                      samples=np.int64(SAMPLES),batch=np.int64(BATCH),block_size=np.int64(BLOCK_SIZE),
                      rank=np.int64(RANK),seed=np.uint64(SEED),seed_xor=np.uint64(xor),
                      rng_seed=np.uint64(RNG_SEEDS[args.stream]))
        with out.open('xb') as handle:
            np.savez(handle,**arrays)
        receipt = dict(schema='cross-depth-angular-energy-acquisition-v1',stream=args.stream,
                       passed=passed,qualifications=qualifications,rank=RANK,pre_layer=13,
                       energy_post_layers=[4,8],samples=SAMPLES,batch=BATCH,batches=batches,
                       block_size=BLOCK_SIZE,blocks=BLOCKS,bin_counts=bin_counts.tolist(),
                       legacy_byte_parity=legacy_parity,legacy_max_abs_difference=legacy_max,
                       cpu_legacy_third_relative=relative_third,
                       cpu_legacy_third_max_abs=float(np.max(np.abs(difference))),
                       empirical_algebra=algebra,norm_membership_mismatch=norm_mismatch.tolist(),
                       conditional_variance_minimum=float(np.min(conditional['variance'])),
                       source_and_input_provenance=provenance,statistics_certificate=small,
                       output_sha256=sha256(out),utc_finished=utc_now(),
                       wall_seconds=time.time()-started,peak_rss_bytes=peak,nice=nice,
                       numpy_version=np.__version__,torch_version=torch.__version__,
                       qualification='Fixed independent8M replay; scalar-bin targets only. '
                       'Actual f32 forward, CPU f64 new moments. No estimator, fit, panel, '
                       'conditional full-covariance source, or score. Shared arithmetic bias remains.')
        with receipt_path.open('x',encoding='utf-8') as handle:
            json.dump(receipt,handle,indent=2,allow_nan=False)
            handle.write('\n')
        record(dict(event='complete',stream=args.stream,passed=passed,
                    output=str(out),output_sha256=receipt['output_sha256'],qualifications=qualifications))
        if not passed:
            raise RuntimeError('saved acquisition failed its frozen qualification; stop interpretation')
    except BaseException as exc:
        failure = dict(utc=utc_now(),exception_type=type(exc).__name__,message=str(exc),
                       traceback=traceback.format_exc(),target_access_started=target_access_started,
                       wall_seconds=time.time()-started)
        with failure_path.open('x',encoding='utf-8') as handle:
            json.dump(failure,handle,indent=2)
            handle.write('\n')
        raise


if __name__ == '__main__':
    main()
