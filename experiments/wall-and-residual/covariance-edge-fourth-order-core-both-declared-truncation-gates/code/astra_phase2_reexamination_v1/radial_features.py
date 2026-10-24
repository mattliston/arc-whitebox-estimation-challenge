"""Target-blind node features for the new radial K3 card, before correction."""
import numpy as np

FEATURE_NAMES = (
    "depth","logical_rank","retained_fraction","alpha","centered_log_sigma",
    "gate","pdf","gaussian_mean_scaled","host_mean_scaled","edgeworth_delta_scaled",
    "pre_skew","post_variance_ratio","pre_kurtosis","rho_mean","rho_rms","rho_abs",
    "edge_w","edge_w2","edge_w3","edge_w4","post_skew","post_kurtosis",
    "k21_row_rms","k21_col_rms","k21_row_mean","k21_col_mean",
    "rho_alpha","post_cov_row_rms","post_cov_row_mean","host_global_mean")

def features(l,m,C,K21,aux,W):
    n = len(m); s = aux["sigma"]; v = s*s
    rho = aux["pre_cov"]/np.outer(s,s)
    skplane = K21/(v[:,None]*s[None,:])
    postscale = np.sqrt(np.maximum(aux["post_v"],1e-12))
    postrho = C/np.outer(postscale,postscale)
    postrho = np.clip(postrho,-1,1)
    w = W.astype(np.float32); w2 = w*w; w3 = w2*w; w4 = w2*w2
    ones = np.ones(n)
    vals = (ones*(2*l/15-1),ones*(l/16),ones,aux["alpha"],np.log(s)-np.mean(np.log(s)),
            aux["gate"],aux["pdf"],aux["gaussian_mean"]/s,m/s,(m-aux["gaussian_mean"])/s,
            aux["pre_k3"]/(v*s),aux["post_v"]/v,aux["pre_k4"]/(v*v),np.mean(rho,axis=1),
            np.sqrt(np.mean(rho*rho,axis=1)),np.mean(np.abs(rho),axis=1),
            np.sum(w,axis=0)*0.7071067811865475,np.sum(w2-0.001953125,axis=0)*11.313708498984761,
            np.sum(w3,axis=0)*93.47798314754834,np.sum(w4-1.1444091796875e-5,axis=0)*836.0924988699916,
            aux["post_k3"]/(v*s),aux["post_k4"]/(v*v),
            np.sqrt(np.sum(skplane*skplane,axis=1)),np.sqrt(np.sum(skplane*skplane,axis=0)),
            np.sum(skplane,axis=1)/np.sqrt(n),np.sum(skplane,axis=0)/np.sqrt(n),
            rho@aux["alpha"]/n,np.sqrt(np.mean(postrho*postrho,axis=1)),np.mean(postrho,axis=1),
            ones*np.mean(m/s))
    return np.stack(vals,axis=1).astype(np.float32)
