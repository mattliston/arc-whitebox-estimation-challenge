"""Frozen energy-bin sufficient statistics and consumers, NumPy only.

No data, target, estimator, or Torch import at module import time.
Used by the reviewed acquisition and independent CPU judge.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WIDTH = 1024
SAMPLES = 8_000_000
BATCH = 65_536
BLOCK_SIZE = 1_000_000
BLOCKS = 8
BINS = 4
CPU_CHUNK = 8192
RANK = 35
SEED = 2758000291903877946
XORS = {'A': 0x4F4E454849474831, 'B': 0x4F4E454849474832}
RNG_SEEDS = {'A': 7568343082859363083, 'B': 7568343082859363080}
THRESHOLDS = np.array([1152.5507888832508, 1166.1889834731694], np.float64)
ROW = HERE / 'rows_current_transport_shadow_v2/row_0035'
STORE = HERE.parent / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
WEIGHT_ROW = STORE / 'rows/observer_fit/0035'
BANK_INPUT = HERE / 'fourth_trace_oracle_inputs_r35_v1.npz'
SHIFT_INPUT = HERE / 'residual_one_high_inputs_r35_l12_v1.npz'
CERTIFICATE = HERE / 'cross_depth_energy_total_cumulance_certificate_v1.json'
PROTOCOL_NAMES = tuple(f'CROSS_DEPTH_ANGULAR_ENERGY_INFORMATION_PROTOCOL_V{v}.md' for v in range(1, 5))
PRIOR_HASHES = {
    'A': '23460f0205733e56a37880ebcfab824b20986acc1067f0645eaa03dfe332ad66',
    'B': '81d97d2f4d268561eb56ac6bec9d97b26b061ece77cc6094aa098619f0ef14ad',
}
INPUT_HASHES = {
    ROW / 'x.npy': '1b73084ccdd4c0f72b0bb56f78b5454dc0e9d6d6340d4ae851af956e59660dce',
    ROW / 'sigma.npy': '0b3a141b31a7f3a1e66171986b60d6a789ad865c01ab63d9821505fb8f851ca9',
    ROW / 'host.npy': '9edcb7af6bdba5f40f65302a3abfd3f1a13219622fbf4016b627b19c4a71fa7e',
    ROW / 'baseline.npy': 'b5c1fa2014bb9d0c80b960253afa2ad266a2bd8af26045d371c8fb3f28ff5181',
    ROW / 'result.json': '11dbf5f9e77f8b181d9cf992cc6d92da73f0043c8981715404f77f8ac17c3770',
    WEIGHT_ROW / 'row_manifest.json': '5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809',
    WEIGHT_ROW / 'weights.npy': '14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2',
    BANK_INPUT: '2666281e0bc5c95a5efa74d851179187b1cb2b3350ef1566965000dd4256a3e8',
    SHIFT_INPUT: '512c1bbbc3498d1cf02a7b7ad9159416b75a41a8fba0363d20b8e9c40c463efd',
    HERE / 'acquire_residual_one_high_oracle_v1.py': 'd9d7f23e2c59cdf73815a99e1e323d8afd50c29ed375110809d3dbb04af2e8fa',
    HERE / 'payload_compact_feed_fullpair_v1.npz': '166b3c67ec4d27793d433e38e15086b95d052426fdf435a79067cc07a2a4dd7e',
    HERE / 'radial_features.py': '6fda3eeb3dfca19fdf60bd44b4f427f8dbd016542af53ec1e721efafe43fd30e',
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path)
    stat = path.stat()
    return dict(path=str(path), sha256=sha256(path), bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat())


def verify_frozen_inputs():
    records = {}
    for path, expected in INPUT_HASHES.items():
        item = file_record(path)
        if item['sha256'] != expected:
            raise ValueError(f'frozen input digest mismatch: {path}')
        records[str(path)] = item
    manifest = json.loads((WEIGHT_ROW/'row_manifest.json').read_text())
    if (manifest['role'] != 'observer_fit' or manifest['role_rank'] != RANK
            or int(manifest['mlp_seed']) != SEED
            or manifest['files']['weights.npy']['sha256'] != INPUT_HASHES[WEIGHT_ROW/'weights.npy']):
        raise ValueError('authenticated weight manifest identity mismatch')
    x = np.load(ROW/'x.npy', mmap_mode='r')
    s = np.load(ROW/'sigma.npy', mmap_mode='r')
    h = np.load(ROW/'host.npy', mmap_mode='r')
    thresholds = np.array([np.sum(s[l].astype(np.float64)**2*x[l,:,11].astype(np.float64)
                                 + h[l].astype(np.float64)**2, dtype=np.float64)
                           for l in (4, 8)])
    if not np.array_equal(thresholds, THRESHOLDS):
        raise ValueError(f'frozen threshold reconstruction mismatch: {thresholds}')
    cert = json.loads(CERTIFICATE.read_text())
    if not cert['algebra']['passed'] or cert['process_exit_code'] != 0:
        raise ValueError('root total-cumulance certificate did not pass')
    for line in cert['source_and_protocol_sha256_output'].splitlines():
        expected, name = line.split(maxsplit=1)
        if sha256(HERE/name) != expected:
            raise ValueError(f'root certificate source/protocol mismatch: {name}')
    for name in PROTOCOL_NAMES:
        records[str(HERE/name)] = file_record(HERE/name)
    records[str(CERTIFICATE)] = file_record(CERTIFICATE)
    return records


def norm_sums(values):
    """Only the reduction dtype changes in the secondary rounding counter."""
    return (np.einsum('ij,ij->i', values, values, dtype=np.float64, casting='unsafe'),
            np.einsum('ij,ij->i', values, values, dtype=np.float32).astype(np.float64))


class BlockSums:
    """Four shifted raw sums per bin/block plus independent global sums."""
    def __init__(self, shift, block_size=BLOCK_SIZE, blocks=BLOCKS):
        self.shift = np.asarray(shift, dtype=np.float64)
        self.block_size = int(block_size)
        self.blocks = int(blocks)
        self.counts = np.zeros((blocks, BINS), np.int64)
        self.sums = np.zeros((blocks, BINS, 4, len(self.shift)), np.float64)
        self.global_sums = np.zeros((4, len(self.shift)), np.float64)
        self.samples = 0

    @staticmethod
    def raw_sums(z, shift):
        y = z.astype(np.float64)-shift
        y2 = y*y
        return np.stack((np.sum(y, axis=0, dtype=np.float64),
                         np.sum(y2, axis=0, dtype=np.float64),
                         np.sum(y2*y, axis=0, dtype=np.float64),
                         np.sum(np.maximum(z, 0), axis=0, dtype=np.float64)))

    def update(self, z, memberships):
        if z.ndim != 2 or z.shape[1] != len(self.shift) or len(memberships) != len(z):
            raise ValueError('invalid sufficient-statistic input shapes')
        if np.any(memberships < 0) or np.any(memberships >= BINS):
            raise ValueError('invalid frozen membership')
        offset = 0
        while offset < len(z):
            block = self.samples // self.block_size
            if block >= self.blocks:
                raise ValueError('more samples than the frozen block count')
            count = min(CPU_CHUNK, len(z)-offset,
                        self.block_size-self.samples % self.block_size)
            zz = z[offset:offset+count]
            bb = memberships[offset:offset+count]
            self.global_sums += self.raw_sums(zz, self.shift)
            for b in range(BINS):
                selected = zz[bb == b]
                self.counts[block, b] += len(selected)
                if len(selected):
                    self.sums[block, b] += self.raw_sums(selected, self.shift)
            offset += count
            self.samples += count


def moments(counts, sums, shift):
    """Finite-population empirical moments AND unbiased k-statistics."""
    count = np.asarray(counts, dtype=np.float64)
    if np.any(count < 3):
        raise ValueError('fewer than three samples in a moment cell')
    raw = np.asarray(sums, dtype=np.float64)/count[...,None,None]
    ybar = raw[...,0,:]
    m2 = raw[...,1,:]-ybar*ybar
    m3 = raw[...,2,:]-3*ybar*raw[...,1,:]+2*ybar*ybar*ybar
    return dict(mean=np.asarray(shift)+ybar, mean_shifted=ybar,
                variance_empirical=m2, third_empirical=m3,
                variance=m2*(count/(count-1))[...,None],
                third=m3*(count*count/((count-1)*(count-2)))[...,None],
                true_postmean=raw[...,3,:])


def scalar_empirical_algebra(counts, sums, global_sums, shift):
    """Checks with biased moments; intentionally does not use k-statistics."""
    bc = np.sum(counts,axis=0)
    bs = np.sum(sums,axis=0)
    cond = moments(bc,bs,shift)
    total = moments(int(np.sum(bc)),global_sums,shift)
    p = bc/np.sum(bc)
    d = cond['mean_shifted']-total['mean_shifted']
    mean_reconstructed = np.einsum('b,bi->i',p,cond['mean_shifted'])
    variance_terms = p[:,None]*(cond['variance_empirical']+d*d)
    third_terms = p[:,None]*(cond['third_empirical']+3*d*cond['variance_empirical']+d**3)
    pairs = dict(mean=(mean_reconstructed,total['mean_shifted']),
                 variance=(np.sum(variance_terms,axis=0),total['variance_empirical']),
                 third=(np.sum(third_terms,axis=0),total['third_empirical']),
                 true_postmean=(np.einsum('b,bi->i',p,cond['true_postmean']),total['true_postmean']))
    errors = {name:float(np.max(np.abs(a-b))/max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(b)))))
              for name,(a,b) in pairs.items()}
    errors['raw_partition'] = float(np.max(np.abs(np.sum(bs,axis=0)-global_sums))
                                   / max(1.,float(np.max(np.abs(global_sums)))))
    return errors


def gaussian_relu(mean, variance):
    # Imported only by CPU judgment/certification, never by MPS acquisition.
    from scipy.special import ndtr
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    if np.any(variance < 0) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
        raise ValueError('invalid conditional Gaussian moment, no repair allowed')
    result = np.maximum(mean,0)  # Exact degenerate-Gaussian limit, not a variance floor.
    positive = variance > 0
    sigma = np.sqrt(variance[positive])
    alpha = mean[positive]/sigma
    result[positive] = mean[positive]*ndtr(alpha)+sigma*np.exp(-.5*alpha*alpha)/np.sqrt(2*np.pi)
    return result


def consumer(counts, sums, shift):
    count = np.sum(counts,axis=0) if np.ndim(counts) == 2 else np.asarray(counts)
    summed = np.sum(sums,axis=0) if np.ndim(sums) == 4 else np.asarray(sums)
    conditional = moments(count,summed,shift)
    p = count/np.sum(count)
    mixture = np.einsum('b,bi->i',p,gaussian_relu(conditional['mean'],conditional['variance']))
    total = moments(int(np.sum(count)),np.sum(summed,axis=0),shift)
    return dict(G=mixture,G0=gaussian_relu(total['mean'],total['variance']),
                true=total['true_postmean'],p=p,conditional=conditional,total=total)


def legacy_finish(post_sum,next_sum,next_second,next_square,next_third,indices,count):
    # Byte-for-byte arithmetic order copied from the corresponding part of
    # acquire_residual_one_high_oracle_v1.finish; no current target is used.
    y=next_sum/count;V=next_second/count;y2=next_square/count
    K=next_third/count-2*y[indices,None]*V-y[None,:]*y2[indices,None]+2*(y[indices]**2)[:,None]*y[None,:]
    unbiased=count*count/((count-1)*(count-2))
    return dict(post_mean_residual=post_sum/count,next_pre_mean_residual=y,next_K21=K*unbiased)


def statistic_certificate():
    """Tiny seeded data only; independently checks the actual new accumulator."""
    rng = np.random.Generator(np.random.PCG64DXSM(90620262014))
    z = rng.standard_normal((256,5)).astype(np.float32)
    z += np.float32(.13)*z*z
    labels = np.arange(len(z),dtype=np.int64)%4
    shift = np.array([.7,-.1,.3,1.1,-.9],np.float32)
    obj = BlockSums(shift,block_size=32)
    # Deliberately cross both synthetic block and update boundaries.
    for start in range(0,len(z),37):
        obj.update(z[start:start+37],labels[start:start+37])
    if not np.array_equal(np.sum(obj.counts,axis=1),np.full(8,32)):
        raise AssertionError('synthetic block segmentation mismatch')
    direct = BlockSums.raw_sums(z,shift.astype(np.float64))
    errors = scalar_empirical_algebra(obj.counts,obj.sums,obj.global_sums,shift)
    errors['direct_raw_sum'] = float(np.max(np.abs(direct-obj.global_sums))
                                     / max(1.,float(np.max(np.abs(direct)))))
    pop = z.astype(np.float64)-z.astype(np.float64).mean(0)
    expected = np.mean(pop**3,axis=0)*len(z)**2/((len(z)-1)*(len(z)-2))
    estimated = moments(len(z),obj.global_sums,shift)['third']
    errors['unconditional_k_statistic'] = float(np.max(np.abs(expected-estimated)))
    if max(errors.values()) > 2e-12:
        raise AssertionError(errors)
    return dict(passed=True,errors=errors,target_access=False,torch_imported=False)
