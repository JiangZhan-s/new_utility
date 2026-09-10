"""Independently recompute the user's two-client KKT example."""
import json
from pathlib import Path
import numpy as np
from .mechanism import instance,solve


def main():
    z=instance(np.array([[140,139,121],[127,128,145]]),steps=2)
    z['s'][:]=.02; z['alpha']=.05/z['lam']; z['beta']=.1/z['lam']**2; z['budget']=.11
    result=solve(z,tol=1e-9)
    cb,ch,cl,ca=z['coef']; E0=float(z['e'][0]@z['e'][0]); kh=ch/cb; kn=ca*z['noise'][0]/(cb*z['g2'])
    u=(-.07+np.sqrt(.07**2+4*.2*.105))/.4
    H=E0/4*np.array([[1+kh,kh-1],[kh-1,1+kh]])
    g=kh*E0/2*np.ones(2); j0=kh*E0+kn/2
    D=.1*np.eye(2); p=.035*np.ones(2); effective=.105; x=np.full(2,u)
    nu=kh*E0*(1-u)/(.035+.2*u)
    M=H+nu*D; zz=-2*g+nu*p
    primal=float(x@H@x-2*g@x+j0)
    dual=float(j0-nu*effective-.25*zz@np.linalg.solve(M,zz))
    # One participant may spend the entire B; full enhancement is too expensive.
    single_u=(-.035+np.sqrt(.035**2+4*.1*(.11-.0025)))/.2
    report=dict(lambda_value=z['lam'].tolist(),E0=E0,kappa_H=kh,kappa_N=kn,
        u=u,q=(z['lam']*u).tolist(),price=1.1*(.05+.2*u),payment_per_client=.055,
        utility_per_client=.5*(.1*u*u+.02*u-.015),budget_multiplier=nu,
        primal_objective=primal,dual_objective=dual,duality_gap=primal-dual,
        stationarity_inf=float(np.max(np.abs(2*M@x-2*g+nu*p))),
        budget_residual=float(x@D@x+p@x-effective),
        complementarity=float(abs(nu*(x@D@x+p@x-effective))),
        single_participant_u=single_u,single_participant_objective=E0*(1-single_u)**2+kn,
        enumeration=result,
        normalization='User J = (code objective_normalized - C_L*v/C_B)/G_cls^2')
    out=Path('outputs/kkt_review'); out.mkdir(parents=True,exist_ok=True)
    (out/'example.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
