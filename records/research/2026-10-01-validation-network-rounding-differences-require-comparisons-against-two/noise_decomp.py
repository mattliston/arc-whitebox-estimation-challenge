"""TAG opus_engineer noise decomposition: why every rounding-level variant reads +0.1..0.2% MSE against RRPI on the screen and check
networks, and what its real cost is.
usage: noise_decomp.py     (reads stored predictions only; no estimator run)
The plain-matmul diagnostic (d01: every fast-multiplication kernel off) is the estimator WITHOUT fast-multiplication rounding noise.
Write any variant's final-layer prediction as  y = y_d01 + n  and the noise-free error as  c = y_d01 - truth.  Then
  MSE(variant) = |c|^2 + 2<c, n> + |n|^2
    |n|^2   = the variant's rounding-noise POWER: a real cost, the same on every network set;
    2<c, n> = how the particular noise draw happens to ALIGN with the error: zero on average for a draw nobody selected, with a
              per-network sd of about 0.24% of the MSE.
Printed per variant, pooled over the 12 networks (screen 35,100,36,45 + check 52..88), in % of the noise-free MSE."""
import numpy as np

P2 = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16'
H = P2 + '/codex_resume_20260926_v2'
E = P2 + '/opus_reexam_20261001_v1/engineer'
SCREEN = (35, 100, 36, 45)
RANKS = [35, 100, 36, 45, 52, 58, 64, 70, 76, 80, 84, 88]


def truth(rank):
    return np.load(P2 + f'/fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)


def native(screen_tag, check_tag):
    return lambda r: np.load(f"{H}/{screen_tag if r in SCREEN else check_tag}/pred_{r}.npy")[15].astype(np.float64)


def quick(name):
    return lambda r: np.load(f'{E}/logs/quick_{name}_pred_{r}.npy')[15].astype(np.float64)


d01 = native('opus_engineer_d01_plain_screen', 'opus_engineer_d01_plain_check')
VARIANTS = [
    ('RRPI (parent)', native('opus_engineer_rrpi_screen_a', 'opus_engineer_rrpi_check_c')),
    ('a04 strips, first version', native('opus_engineer_a04_k250k_g750k_screen', 'opus_engineer_a04_k250k_g750k_check')),
    ('a07 strips, pad-only', native('opus_engineer_a07_k750000_g750000_d50_screen', 'opus_engineer_a07_k750000_g750000_d50_check')),
    ('a07 strips, full', native('opus_engineer_a07_k250000_g750000_d51_screen', 'opus_engineer_a07_k250000_g750000_d51_check')),
    ('stack T+C+B+F+D+Hfull', quick('tcbfd_hfull_binds_12')),
]
if __name__ == '__main__':
    print(f"{'variant':28s} {'noise power':>12s} {'alignment':>10s} {'(se)':>7s} {'MSE vs noise-free':>18s} {'MSE vs RRPI':>12s}")
    base = None
    for name, load in VARIANTS:
        c2 = cross = n2 = 0.0
        per = []
        for r in RANKS:
            t = truth(r); yd = d01(r); y = load(r)
            c = yd - t; n = y - yd
            c2 += np.mean(c * c); cross += 2 * np.mean(c * n); n2 += np.mean(n * n)
            per.append(100 * 2 * np.mean(c * n) / np.mean(c * c))
        per = np.array(per)
        total = 100 * (cross + n2) / c2
        base = total if base is None else base
        print(f'{name:28s} {100*n2/c2:+11.3f}% {100*cross/c2:+9.3f}% {per.std(ddof=1)/np.sqrt(len(RANKS)):6.3f}% {total:+17.3f}% {100*((1+total/100)/(1+base/100)-1):+11.3f}%')
