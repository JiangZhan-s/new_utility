import unittest
import numpy as np
from baqp.mechanism import instance,solve,response,coefficients,objective,prices

class MechanismTests(unittest.TestCase):
    def interior(self):
        z=instance(np.array([[400,0],[0,400]]),steps=2)
        z['s'][:]=.02; z['alpha']=.05/z['lam']; z['beta']=.1/z['lam']**2
        z['budget']=.11
        return z

    def test_analytic_interior(self):
        z=self.interior(); s=solve(z,tol=1e-9)
        np.testing.assert_allclose(s['u'],[.5704025758]*2,atol=2e-6)
        np.testing.assert_allclose(s['payment'],[.055]*2,atol=1e-8)
        self.assertLess(s['normalized_absolute_gap'],1e-7)

    def test_independent_grid(self):
        z=self.interior(); s=solve(z)
        # Independent price-space search tests implementation and lower bound.
        r=np.linspace(0,.35,301); best=np.inf
        for a in r:
            for b in r:
                price=np.array([a,b]); active,u,_=response(z,price)
                pay=z['d']@ (price*(1-z['lam']/z['bar']+u*z['lam']/z['bar']))
                if active.any() and pay<=z['budget']:
                    best=min(best,objective(z,active,u))
        self.assertLessEqual(s['normalized_lower_bound'],best+1e-8)
        self.assertLessEqual(s['objective_normalized'],best+1e-6)

    def test_prices_and_deviation(self):
        z=self.interior(); sol=solve(z)
        qgrid=np.linspace(0,1,10001)
        for k,r in enumerate(sol['price']):
            q=qgrid*z['lam'][k]
            utility=r*(1-z['lam'][k]/z['bar']+q/z['bar'])-z['s'][k]-z['alpha'][k]*q-z['beta'][k]*q*q
            self.assertAlmostEqual(qgrid[utility.argmax()],sol['u'][k],places=3)
        # Discrete clients have a different full-quality price threshold.
        cont=prices(z,np.ones(2,dtype=bool),np.ones(2))
        disc=prices(z,np.ones(2,dtype=bool),np.ones(2),True)
        self.assertTrue(np.all(cont>disc))

    def test_zero_heterogeneity_and_infeasible(self):
        z=instance(np.array([[20,20],[30,30]]),budget_fraction=1)
        s=solve(z); self.assertEqual(s['status'],'ok')
        self.assertEqual(s['u'],[0.,0.])
        z['budget']=0; self.assertEqual(solve(z)['status'],'infeasible')

    def test_discrete_tie_break(self):
        # Values that exposed cancellation in the real CIFAR-100/FMNIST run.
        for seed in range(20):
            rng=np.random.default_rng(seed)
            z=instance(rng.integers(1,100,(6,10)),seed=seed,budget_fraction=2)
            threshold=z['bar']*(z['alpha']+z['beta']*z['lam'])
            active,u,_=response(z,threshold,discrete=True)
            np.testing.assert_array_equal(u[active],np.ones(active.sum()))
            result=solve(z,discrete=True)
            self.assertEqual(result['status'],'ok')

    def test_user_kkt_example(self):
        z=instance(np.array([[140,139,121],[127,128,145]]),steps=2)
        z['s'][:]=.02; z['alpha']=.05/z['lam']; z['beta']=.1/z['lam']**2; z['budget']=.11
        result=solve(z,tol=1e-9)
        np.testing.assert_allclose(z['lam'],.08485281374238572,atol=1e-14)
        np.testing.assert_allclose(result['u'],.5704025758,atol=2e-6)
        cb,ch,cl,ca=z['coef']; E0=float(z['e'][0]@z['e'][0])
        kh=ch/cb; kn=ca*z['noise'][0]/(cb*z['g2'])
        u=np.array(result['u']); nu=kh*E0*(1-u[0])/(.035+.2*u[0])
        scalar_j=(result['objective_normalized']-cl*z['noise'][0]/cb)/z['g2']
        self.assertAlmostEqual(scalar_j,2.61828910e-5,places=11)
        self.assertAlmostEqual(nu,3.91703334e-5,places=10)
        self.assertAlmostEqual(kn,.00004734848484848486,places=14)

    def test_protocol_coefficients(self):
        np.testing.assert_allclose(coefficients(),[217.712669,26.444138,8.788034,.989603],rtol=1e-6)
        self.assertEqual(coefficients(steps=1)[1],0)
        self.assertEqual(coefficients(steps=1)[2],0)

if __name__=='__main__': unittest.main()
