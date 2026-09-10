"""Independent KKT sanity check for the two-client two-part-tariff example."""
import json
from pathlib import Path
import numpy as np
from .mechanism import instance,solve,implementation_payment


def main():
    z=instance(np.array([[140,139,121],[127,128,145]]),steps=2)
    z['s'][:]=.02; z['alpha']=.05/z['lam']; z['beta']=.1/z['lam']**2; z['budget']=.11
    result=solve(z,tol=1e-9)
    active=np.asarray(result['active'],dtype=bool); u=np.asarray(result['u'],dtype=float)
    payment=implementation_payment(z,active,u)
    report=dict(
        lambda_value=z['lam'].tolist(),
        u=u.tolist(),
        q=(z['lam']*u).tolist(),
        fixed_transfer=result['fixed_transfer'],
        marginal_price=result['marginal_price'],
        payment=result['payment'],
        direct_cost_payment=payment.tolist(),
        max_payment_cost_error=float(np.max(np.abs(np.asarray(result['payment'])-payment))),
        client_utility=result['client_utility'],
        objective_normalized=result['objective_normalized'],
        normalized_lower_bound=result['normalized_lower_bound'],
        normalized_absolute_gap=result['normalized_absolute_gap'],
        budget=z['budget'],
        total_payment=result['total_payment'],
        interpretation='Signed fixed transfer + marginal quality price; IR binds and total payment equals true client cost under complete information.',
        enumeration=result)
    out=Path('outputs/kkt_review_two_part'); out.mkdir(parents=True,exist_ok=True)
    (out/'example.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
