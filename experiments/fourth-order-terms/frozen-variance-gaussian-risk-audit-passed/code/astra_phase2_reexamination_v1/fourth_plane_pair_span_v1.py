"""Input-only pair basis and deterministic matrix-free ridge solve."""
import numpy as np


def input_basis(f):
    a, p, g, s = (f['pre13_'+k] for k in ('alpha', 'pdf', 'gate', 'sigma'))
    skew = np.diag(f['pre13_K21'])/s**3
    kurt = f['pre13_k4']/s**4
    ms = f['pre13_m']/s
    F = np.stack((g, p, a*p, (a*a-1)*p, ms, a*g, skew*g, skew*p,
                  skew*a*p, skew*ms, kurt*g, kurt*p, kurt*a*p, kurt*ms), axis=1)
    U, singular, Vt = np.linalg.svd(F, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0]*1e-12))
    H = U[:, :rank]*np.sqrt(len(F))
    reconstruction = H@(H.T@F)/len(F)
    error = float(np.linalg.norm(reconstruction-F)/np.linalg.norm(F))
    assert error <= 1e-10, error
    R = f['pre13_Cpre']/np.outer(s, s)
    R = (R+R.T)/2
    powers = np.stack((np.ones_like(R), R, R*R))
    for power in powers:
        np.fill_diagonal(power, 0)
    rms = np.sqrt(np.mean(powers*powers, axis=(1, 2)))
    assert rms.min() > 0
    powers /= rms[:, None, None]
    return H, powers, dict(numerical_rank=rank, original_features=14,
                          singular_values=singular.tolist(), span_reconstruction_relative=error,
                          correlation_power_rms=rms.tolist())


class PlaneOperator:
    def __init__(self, H, powers, symmetric, ridge=.001):
        self.H, self.powers = H, powers
        self.symmetric, self.ridge = symmetric, ridge
        self.shape = (len(powers), H.shape[1], H.shape[1])
        self.scale = float(len(H)**2)
        square = H*H
        self.preconditioner = np.stack([square.T@(r*r)@square/self.scale+ridge for r in powers])

    def project(self, theta):
        return (theta+theta.transpose(0, 2, 1))/2 if self.symmetric else theta

    def forward(self, theta):
        theta = self.project(theta)
        return sum(r*((self.H@t)@self.H.T) for r, t in zip(self.powers, theta))

    def adjoint(self, matrix):
        return self.project(np.stack([self.H.T@(r*matrix)@self.H for r in self.powers]))

    def normal(self, theta):
        return self.adjoint(self.forward(theta))/self.scale+self.ridge*theta

    def fit(self, target, tolerance=1e-7, max_iterations=300):
        rhs = self.adjoint(target)/self.scale
        norm = float(np.linalg.norm(rhs))
        assert norm > 0
        x = np.zeros_like(rhs)
        residual = rhs.copy()
        z = residual/self.preconditioner
        direction = z.copy()
        rz = float(np.sum(residual*z))
        history = []
        for iteration in range(1, max_iterations+1):
            action = self.normal(direction)
            denominator = float(np.sum(direction*action))
            assert denominator > 0, denominator
            step = rz/denominator
            x += step*direction
            residual -= step*action
            relative = float(np.linalg.norm(residual)/norm)
            if iteration%25 == 0 or relative <= tolerance:
                history.append(dict(iteration=iteration, recursive_relative_residual=relative))
            if relative <= tolerance:
                break
            z = residual/self.preconditioner
            rz_new = float(np.sum(residual*z))
            direction = z+(rz_new/rz)*direction
            rz = rz_new
        actual = float(np.linalg.norm(self.normal(x)-rhs)/norm)
        record = dict(iterations=iteration, relative_normal_residual=actual,
                      passed=actual <= tolerance, ridge=self.ridge, history=history)
        return x, record
