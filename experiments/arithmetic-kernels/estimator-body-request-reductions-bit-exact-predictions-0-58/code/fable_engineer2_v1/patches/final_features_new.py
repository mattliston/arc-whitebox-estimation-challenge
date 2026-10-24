def final_scalar_features(l,m,aux,W):
    n = len(m); s = aux["sigma"]; v = s*s
    c = _e2_feature_columns(n, l); ones = c[2]; zero = fnp.zeros(n)
    ws = _e2_wsums(_e2_wpow(W))
    ls = fnp.log(s); ms = m/s; vs = v*s; vsq = v*v
    vals = (c[0],c[1],ones,aux["alpha"],ls-fnp.mean(ls),
            aux["gate"],aux["pdf"],aux["gaussian_mean"]/s,ms,(m-aux["gaussian_mean"])/s,
            aux["pre_k3"]/vs,aux["post_v"]/v,aux["pre_k4"]/vsq,zero,zero,zero,
            ws[0],ws[1],ws[2],ws[3],
            aux["post_k3"]/vs,aux["post_k4"]/vsq,zero,zero,zero,zero,zero,zero,zero,
            ones*fnp.mean(ms))
    return fnp.stack(vals,axis=1,dtype=fnp.float32)

