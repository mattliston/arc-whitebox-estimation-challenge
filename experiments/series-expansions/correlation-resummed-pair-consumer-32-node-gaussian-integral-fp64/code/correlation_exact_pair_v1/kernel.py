"""FP64 Gaussian boundary-moment consumer; fnp/CDF/GL globals are injected."""

def _ec_boundary(x,y,pdfx,rho,sd,den):
    c=y-rho*x;z=c/sd;pc=fnp.exp(-z*z/2)*.3989422804014327;gc=_ec_cdf(z)
    moments=[gc,sd*pc+c*gc,(c*c+den)*gc+c*sd*pc]
    first=[pc/sd,gc,2*moments[1]]
    second=[-z*pc/den,pc/sd,2*gc]
    boundary={}
    for k in range(3):
        boundary[(k,0)]=pdfx*moments[k]
        boundary[(k,1)]=-pdfx*(x*moments[k]+rho*first[k])
        boundary[(k,2)]=pdfx*((x*x-1)*moments[k]+2*x*rho*first[k]+rho*rho*second[k])
    return boundary

def _ec_geometry(a, rho, pdf, g):
    if not bool(fnp.all(fnp.isfinite(rho))) or not bool(fnp.all(fnp.abs(rho) <= .85)):
        raise ValueError('correlation consumer outside fixed |rho|<=.85 domain')
    x=a[:,None];y=a[None,:]
    aa=x*x+y*y;ab=2*x*y
    total=fnp.zeros_like(rho,dtype=fnp.float64)
    nodes=fnp.asarray(_ec_nodes,dtype=fnp.float64)
    weights=fnp.asarray(_ec_weights,dtype=fnp.float64)
    for start in range(0,32,8):
        t=((nodes[start:start+8]+1)/2)[:,None,None]*rho[None,:,:]
        den=1-t*t
        density=fnp.exp(-.5*(aa[None,:,:]-t*ab[None,:,:])/den)/(6.283185307179586*fnp.sqrt(den))
        total=total+fnp.sum(weights[start:start+8,None,None]*density,axis=0)
    P=g[:,None]*g[None,:]+rho*total/2
    den=1-rho*rho;sd=fnp.sqrt(den)
    # Keep the literal rho_ij in both boundary orientations: native C may
    # contain roundoff antisymmetry. No symmetrization is introduced here.
    bx=_ec_boundary(x,y,pdf[:,None],rho,sd,den)
    by=_ec_boundary(y,x,pdf[None,:],rho,sd,den)
    joint=fnp.exp(-.5*(aa-rho*ab)/den)/(6.283185307179586*sd)
    M={(0,0):P}
    M[(1,0)]=x*P+bx[(0,0)]+rho*by[(0,0)]
    M[(0,1)]=y*P+by[(0,0)]+rho*bx[(0,0)]
    M[(2,0)]=x*M[(1,0)]+P+rho*by[(1,0)]
    M[(0,2)]=y*M[(0,1)]+P+rho*bx[(1,0)]
    M[(1,1)]=x*M[(0,1)]+rho*P+bx[(1,0)]
    M[(2,1)]=x*M[(1,1)]+M[(0,1)]+rho*M[(1,0)]
    M[(1,2)]=y*M[(1,1)]+M[(1,0)]+rho*M[(0,1)]
    M[(2,2)]=x*M[(1,2)]+M[(0,2)]+2*rho*M[(1,1)]
    return M,bx,by,joint

def _ec_derivative(A,B,p,q,geometry):
    M,bx,by,joint=geometry
    # Falling factorials for the only powers1 and2 in this consumer.
    fA=2 if A==2 and p>0 else 1
    fB=2 if B==2 and q>0 else 1
    if p<=A and q<=B:return (fA*fB)*M[(A-p,B-q)]
    if p>A and q<=B:return (fA*fB)*bx[(B-q,p-A-1)]
    if p<=A and q>B:return (fA*fB)*by[(A-p,q-B-1)]
    assert A==1 and B==1 and p==2 and q==2
    return joint

def _ec_pairs(a,rho,pdf,g,T21,T22,T31,t3,t4):
    geometry=_ec_geometry(a,rho,pdf,g)
    u={1:pdf+a*g,2:(a*a+1)*g+a*pdf}
    d3={1:-a*pdf,2:2*pdf};d4={1:(a*a-1)*pdf,2:-2*a*pdf}
    results=[]
    for A,B in [(1,1),(2,1),(2,2)]:
        def der(p,q):return _ec_derivative(A,B,p,q,geometry)
        out=geometry[0][(A,B)]-u[A][:,None]*u[B][None,:]
        out=out+(T21*der(2,1)+T21.T*der(1,2))/2
        out=out+T22*der(2,2)/4
        out=out+(T31*der(3,1)+T31.T*der(1,3))/6
        out=out+(t3[:,None]*(der(3,0)-d3[A][:,None]*u[B][None,:])+t3[None,:]*(der(0,3)-u[A][:,None]*d3[B][None,:]))/6
        out=out+(t4[:,None]*(der(4,0)-d4[A][:,None]*u[B][None,:])+t4[None,:]*(der(0,4)-u[A][:,None]*d4[B][None,:]))/24
        results.append(out)
    return results
