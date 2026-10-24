"""Target-blind response-weighted contractions of current joint moments."""
import numpy as np

KERNEL_NAMES=("t21","t12","t21_rho","t12_rho","rho2","t21_t12")
FUNCTION_NAMES=("one","gate","pdf","alpha_pdf","h2_pdf","skew","kurtosis","host_scaled")
FEATURE_NAMES=tuple(f"joint_{k}_{f}" for k in KERNEL_NAMES for f in FUNCTION_NAMES)

def joint_features(m,K21,aux):
    s=aux["sigma"];a=aux["alpha"];p=aux["pdf"];v=s*s;n=len(s)
    rho=aux["pre_cov"]/np.outer(s,s)
    t=K21/(v[:,None]*s[None,:])
    rho=rho.copy();t=t.copy()
    np.fill_diagonal(rho,0);np.fill_diagonal(t,0)
    functions=np.stack((np.ones(n),aux["gate"],p,a*p,(a*a-1)*p,
                        aux["pre_k3"]/(v*s),aux["pre_k4"]/(v*v),m/s),axis=1).astype(np.float32)
    kernels=(t,t.T,t*rho,t.T*rho,rho*rho,t*t.T)
    return np.concatenate([kernel.astype(np.float32)@functions/np.sqrt(n) for kernel in kernels],axis=1).astype(np.float32)
