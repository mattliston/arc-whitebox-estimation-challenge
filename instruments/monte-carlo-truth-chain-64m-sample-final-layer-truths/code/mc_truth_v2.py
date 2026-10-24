#!/usr/bin/env python3
"""TAG analyst A13 mc_truth v2 (lead 2026-10-12 14:0xZ): mc_truth.py with ONE change: the per-batch sums stay on the GPU and are copied to the
host once per GROUP of 8 batches (one host sync per 8 batches instead of one per batch); on the host every batch's float32 sums are still converted
to float64 and added one batch at a time in the same order, so the truths are bit-identical to mc_truth.py's by construction (checked on a full
network with bitident_mc_v1.py before any use). Profile: 1.054x. mc_truth.py text follows.
TAG analyst A13: Monte Carlo truth of the post-activation means of all 16 layers of a 1024-wide ReLU network on MPS (GPU).
x ~ N(0, I) (torch.randn on the GPU with its one defective value masked, see draw(); seeded per network and per half),
y_l = relu(y_{l-1} @ W[l]) in float32.
Per batch of BATCH samples and per layer the float32 sums over the batch of (y - ref) and (y - ref)^2 are taken on the GPU (ref = the
mean of a separate pilot batch) and added in float64 on the CPU. Two independent halves A and B of HALF_BATCHES batches each.
Output (atomic write): truth_seed{seed}.npz = mc_post (pooled mean), mc_post_A, mc_post_B, mc_post_var (per-unit variance, pooled),
N, N_half, seed, weights_sha256, rng_seeds, batch.
GPU etiquette: ONE GPU job on the box and the physicist has priority. Between batches the file O/gpu_request is looked for; while it
exists the job stops launching batches, frees its GPU cache and sleeps (STATE = paused), and resumes when the file is gone.
STATE (text file next to this script) is rewritten about once a minute: running / paused, network, batch, samples per second.
modes:
  mc_truth.py bench <batches>                    throughput and GPU memory, public network 35, nothing written
  mc_truth.py cert <rank>                        the same code on a public network (weights from the store) -> cert_rank{rank}.npz
  mc_truth.py prod <first k> [last k]            fresh networks seed = 7,100,000 + k, skipping finished ones, until stopped
                                                 (clean stop: create the file STOP next to this script; it is read before each network)
knobs (env): A13_BATCH (32768: 134k samples/s and 1.1 GB of GPU memory; 65536 gave 118k and 2.1 GB), A13_HALF_BATCHES (978 ->
32,047,104 samples per half, 64,094,208 per network)."""
import os, sys, json, time
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[_k] = '1'   # one CPU thread, set before numpy loads
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; O = HERE.parent.parent; P2 = O.parent
sys.path.insert(0, str(HERE))
import loader
BATCH = int(os.environ.get('A13_BATCH', 32768)); HALF_BATCHES = int(os.environ.get('A13_HALF_BATCHES', 978))
REQ = O / 'gpu_request'; STATE = HERE / 'STATE'; dev = torch.device('mps'); L, n = 16, 1024; GROUP = 8
torch.set_num_threads(1)
_last_state = [0.0]


def state(text, force=False):
    if force or time.time() - _last_state[0] > 55:
        STATE.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {text}\n"); _last_state[0] = time.time()


def yield_gpu(label):
    """Between batches: if the physicist asks for the GPU, wait until the request file is gone. Returns the seconds waited."""
    if not REQ.exists(): return 0.0
    t0 = time.time(); torch.mps.synchronize(); torch.mps.empty_cache()
    state(f'paused (O/gpu_request present) at {label}', force=True); print(json.dumps(dict(event='paused', at=label, utc=time.strftime('%FT%TZ', time.gmtime()))), flush=True)
    while REQ.exists(): time.sleep(5)
    print(json.dumps(dict(event='resumed', at=label, waited_s=round(time.time() - t0, 1))), flush=True)
    return time.time() - t0


def draw(gen, flip=False):
    """BATCH x 1024 standard normals on the GPU. torch 2.7 MPS randn returns exactly -14.372 once in about 2^23 values (its uniform
    hits 0; measured 490 in 4.0e9, always that one value: +2.5e-5 on the variance, -3.9e-4 skew, +0.005 excess kurtosis). Those
    values are set to 0 (a 1.2e-7 point mass). What is left (rng_check.py, 1e11 values, two seeds): mean within 4e-6, variance
    within 2e-6 of 1, skew within 4e-6, excess kurtosis -0.6e-4 to -1.1e-4 (the generator's tail ends at 5.167 sigma). flip
    negates the batch; the truth runs flip every second batch, which makes the sampled input law symmetric by construction."""
    x = torch.randn((BATCH, n), generator=gen, device=dev, dtype=torch.float32)
    x.masked_fill_(x < -7.0, 0.0)
    return x.neg_() if flip else x


def batch_sums(wg, ref, gen, flip):
    x = draw(gen, flip)
    s, q = [], []; y = x
    for l in range(L):
        y = torch.relu_(y @ wg[l])
        d = y - ref[l]
        s.append(d.sum(0)); q.append((d * d).sum(0))
    return torch.stack(s + q)                                       # (32, 1024) float32, left on the GPU (v2)


def run_network(w, tag, rng_base, label):
    """w: (16, 1024, 1024) float32, z = y @ w[l]. Returns the result dict and the seconds of GPU work (pauses excluded)."""
    wg = [torch.from_numpy(np.ascontiguousarray(w[l])).to(dev) for l in range(L)]
    seeds = dict(pilot=rng_base * 10 + 3, A=rng_base * 10 + 1, B=rng_base * 10 + 2)
    gen = torch.Generator(device=dev); gen.manual_seed(seeds['pilot'])
    y = draw(gen); ref = []
    for l in range(L):
        y = torch.relu_(y @ wg[l]); ref.append(y.mean(0))
    ref_np = torch.stack(ref).cpu().numpy().astype(np.float64)
    halves = {}; t0 = time.time(); waited = 0.0
    for h in ('A', 'B'):
        gen = torch.Generator(device=dev); gen.manual_seed(seeds[h]); acc = np.zeros((2 * L, n)); pending = []
        for it in range(HALF_BATCHES):
            waited += yield_gpu(f'{label} half {h} batch {it}/{HALF_BATCHES}')
            pending.append(batch_sums(wg, ref, gen, bool(it & 1)))
            if len(pending) == GROUP or it == HALF_BATCHES - 1:          # v2: one copy per group; float64 adds per batch, in order
                for chunk in torch.stack(pending).cpu().numpy(): acc += chunk.astype(np.float64)
                pending = []
            done = (it + 1 + (HALF_BATCHES if h == 'B' else 0)) * BATCH
            state(f'running {label}: {done / 1e6:.1f}M of {2 * HALF_BATCHES * BATCH / 1e6:.1f}M samples, {done / max(time.time() - t0 - waited, 1e-9) / 1e3:.0f}k samples/s')
        halves[h] = acc
    Nh = HALF_BATCHES * BATCH; N = 2 * Nh
    mA, mB = ref_np + halves['A'][:L] / Nh, ref_np + halves['B'][:L] / Nh
    tot = halves['A'] + halves['B']; mean_c = tot[:L] / N; var = tot[L:] / N - mean_c ** 2
    res = dict(mc_post=ref_np + mean_c, mc_post_A=mA, mc_post_B=mB, mc_post_var=var, N=np.int64(N), N_half=np.int64(Nh), batch=np.int64(BATCH),
               rng_seeds=np.array([seeds['pilot'], seeds['A'], seeds['B']], dtype=np.int64))
    del wg, ref
    return res, time.time() - t0 - waited


def noise_report(res):
    """truth noise per layer: half difference (mean over units of (A - B)^2 / 4) against the prediction var / N."""
    hd = ((res['mc_post_A'] - res['mc_post_B']) ** 2).mean(1) / 4; pr = res['mc_post_var'].mean(1) / float(res['N'])
    return hd, pr


def save(path, res, **extra):
    tmp = path.with_suffix('.tmp.npz'); np.savez(tmp, **res, **extra); os.replace(tmp, path)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'bench':
        STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
        w = np.load(STORE / 'rows/observer_fit/0035/weights.npy'); HALF_BATCHES = int(sys.argv[2])
        res, sec = run_network(w, 'bench', 999, 'bench'); torch.mps.synchronize()
        print(json.dumps(dict(mode='bench', batch=BATCH, samples=int(res['N']), seconds=round(sec, 2), samples_per_s=round(float(res['N']) / sec),
                              mps_current_GB=round(torch.mps.current_allocated_memory() / 2 ** 30, 2), mps_driver_GB=round(torch.mps.driver_allocated_memory() / 2 ** 30, 2))), flush=True)
    elif mode == 'cert':
        rank = int(sys.argv[2])
        STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
        w = np.load(STORE / f'rows/observer_fit/{rank:04d}/weights.npy'); assert w.dtype == np.float32 and w.shape == (16, 1024, 1024)
        res, sec = run_network(w, f'cert{rank}', 900000 + rank, f'certificate, public network {rank}')
        save(HERE / f'cert_rank{rank:04d}.npz', res, rank=np.int64(rank), weights_sha256=np.array(loader.weights_sha256(np.ascontiguousarray(w))))
        hd, pr = noise_report(res)
        print(json.dumps(dict(mode='cert', rank=rank, N=int(res['N']), seconds=round(sec, 1), samples_per_s=round(float(res['N']) / sec),
                              half_diff_noise_x1e9=[round(float(x) * 1e9, 3) for x in hd], predicted_noise_x1e9=[round(float(x) * 1e9, 3) for x in pr], done=True)), flush=True)
        state(f'idle: certificate run on network {rank} finished', force=True)
    elif mode == 'prod':
        k = int(sys.argv[2]); last = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9; made = 0
        while k <= last:
            if (HERE / 'STOP').exists():
                state(f'stopped by the STOP file before network k={k}', force=True); print(json.dumps(dict(event='stopped', before_k=k))); sys.exit(0)
            seed = loader.SEED0 + k
            if loader.truth_path(seed).exists(): k += 1; continue
            yield_gpu(f'network k={k} (before start)')
            w = loader.weights(seed); sha = loader.weights_sha256(w)
            res, sec = run_network(w, f'seed{seed}', seed, f'network k={k} seed={seed}')
            save(loader.truth_path(seed), res, seed=np.int64(seed), weights_sha256=np.array(sha))
            hd, pr = noise_report(res); made += 1
            print(json.dumps(dict(event='network', k=k, seed=seed, N=int(res['N']), seconds=round(sec, 1), samples_per_s=round(float(res['N']) / sec),
                                  final_layer_noise_x1e9=round(float(hd[15]) * 1e9, 3), final_layer_predicted_x1e9=round(float(pr[15]) * 1e9, 3),
                                  layer0_noise_x1e9=round(float(hd[0]) * 1e9, 2), half_diff_over_predicted_all_layers=round(float(hd.sum() / pr.sum()), 3),
                                  utc=time.strftime('%FT%TZ', time.gmtime()))), flush=True)
            if made % 50 == 0:
                done = loader.available(); rate = [json.loads(x) for x in open(HERE / 'prod.log') if x.startswith('{"event": "network"')][-50:] if (HERE / 'prod.log').exists() else []
                sec50 = np.mean([r['seconds'] for r in rate]) if rate else sec; nz = np.mean([r['final_layer_noise_x1e9'] for r in rate]) if rate else float(hd[15]) * 1e9
                with open(O / 'TEAM_LOG.md', 'a') as f:
                    f.write(f"[analyst] {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())} A13 fresh networks: {len(done)} truths ready (seeds {done[0]}-{done[-1]}), "
                            f"{sec50:.0f} s of GPU per network over the last 50, final-layer truth noise {nz:.2f}e-9 (mean-squared, 64M samples). Loader: O/analyst/fresh/loader.py\n")
            k += 1
        state('idle: production loop reached its last k', force=True)
