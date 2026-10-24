# Same exact tail residual actions, evaluated only at NP birth hubs.
# saved=(A,B,Q,S,T,rows,cols), S=Q.T@A, T=S@B.T; B already lambda-scaled.
# rows/cols/hubs are unique original-unit indices. Returned indices address hubs.
def _npa_hub_actions(saved,Vt,gt,hubs,n,hold):
    A,B,Q,S,T,rows,cols=saved
    rh,ri,hr=hold(fnp.intersect1d(rows,hubs,assume_unique=True,return_indices=True))
    th,ti,ht=hold(fnp.intersect1d(cols,hubs,assume_unique=True,return_indices=True))
    nr,nt=int(hr.shape[0]),int(ht.shape[0])
    ev=ge=None
    if nr:
        Ah=hold(A[ri]);Qh=hold(Q[ri])
        b=hold(hold(B.T)@hold(Vt[cols]))
        ev=hold(hold(Ah@b)-hold(Qh@hold(S@b)))
    if nt:
        Bh=hold(B[ti]);y=hold(gt[rows])
        ye=hold(hold(y@A)-hold(hold(y@Q)@S))
        ge=hold(ye@hold(Bh.T))
    if nr and nt:
        shared,ir,it=hold(fnp.intersect1d(rh,th,assume_unique=True,return_indices=True))
        if int(shared.shape[0]):
            exact=hold(fnp.einsum('ij,ij->i',hold(Ah[ir]),hold(Bh[it]),optimize=False))
            represented=hold(fnp.einsum('ij,ji->i',hold(Qh[ir]),hold(T[:,hold(ti[it])]),optimize=False))
            diagonal=hold(exact-represented)
            hold(fnp.put(ev,ir,hold(hold(ev[ir])-hold(diagonal*hold(Vt[shared])))))
            hold(fnp.put(ge,it,hold(hold(ge[it])-hold(diagonal*hold(gt[shared])))))
    return hr,ev,ht,ge
