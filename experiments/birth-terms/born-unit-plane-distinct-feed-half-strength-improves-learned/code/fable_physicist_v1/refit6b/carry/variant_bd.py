"""TAG fable-physicist (RF6-13 helper): importable torch copy of the parent zero.py _bd_update (identical to carry/variant_eval.py make_bd;
bit-exact vs the card with mode 'orth'), with a switch that puts the born-unit-side leg direction j1 at PHYSICAL form: the full,
unorthogonalized -(g_i/2) eta_a K21[a,i] at coefficient j1_coef, in place of the learned c1 * orth(dir1)."""


def make_bd(torch, j1_mode='orth', j1_coef=1.0):
    def diag0(x): return torch.diagonal_scatter(x, torch.zeros(x.shape[0], dtype=x.dtype))
    def _bd_update(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):
        n = int(bp.shape[0]); c = coefficients.to(torch.float64)
        alpha, sigma, g = aux['alpha'], aux['sigma'], aux['gate']; eta = alpha / sigma
        half_d = 0.5 * d; negative_half_g = -0.5 * g; negative_half_d = -half_d
        skew = aux['pre_k3'] / (sigma * sigma * sigma); gate_jet = skew * (alpha * alpha - 1) * aux['pdf'] / 6
        denominator = torch.clamp_min(torch.sum(bp * bp, dim=0), 1e-20)
        change = torch.zeros((n, n), dtype=torch.float64)
        for j in range(4):
            if j == 0: direction = half_d[:, None] * K21
            elif j == 1: direction = (negative_half_g[:, None] * K21.T) * eta[None, :]
            elif j == 2: direction = (negative_half_d[:, None] * (Cpre * Cpre)) * eta[None, :]
            else: direction = Cpre * gate_jet[:, None]
            direction = diag0(direction)
            if j == 1 and j1_mode == 'phys':
                change = change + direction * j1_coef
            else:
                projection = torch.sum(bp * direction, dim=0) / denominator
                direction = direction - bp * projection[None, :]
                change = change + direction * c[j]
        change = diag0(change)
        old32 = (bp * alive[:, None]).to(torch.float32); new32 = ((bp + change) * alive[:, None]).to(torch.float32)
        old = old32.to(torch.float64); new = new32.to(torch.float64); difference = new - old
        a64 = a.to(torch.float32).to(torch.float64)
        t64 = (t * alive).to(torch.float32).to(torch.float64); s64 = (s * alive).to(torch.float32).to(torch.float64)
        weight = (a64 * (t64 - s64)) * births; quadratic_weight = (a64 * (t64 * t64 - s64 * s64)) * births
        delta = ((2 * old * difference + difference * difference) * weight[None, :] + quadratic_weight[:, None] * difference.T)
        return new32, diag0(delta)
    return _bd_update
