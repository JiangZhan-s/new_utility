import unittest
import numpy as np
from baqp.mechanism import (instance,solve,response,coefficients,objective,tariffs,
                            implementation_payment,calibrated_types)
from baqp.data import additive_target_counts
from baqp.experiment import local_epoch_steps

class MechanismTests(unittest.TestCase):
    def interior(self):
        z=instance(np.array([[400,0],[0,400]]),steps=2)
        z['s'][:]=.02; z['alpha']=.05/z['lam']; z['beta']=.1/z['lam']**2
        z['budget']=.11
        return z

    def test_data_driven_type_calibration(self):
        counts=np.array([[90,10],[50,50],[180,20]])
        z1=instance(counts,seed=1)
        z2=instance(counts,seed=999)
        # The economic type is now data-driven rather than seed-driven.
        np.testing.assert_allclose(z1['s'],z2['s'])
        np.testing.assert_allclose(z1['alpha'],z2['alpha'])
        np.testing.assert_allclose(z1['beta'],z2['beta'])
        self.assertEqual(z1['type_calibration'],
                         'deterministic_additive_missing_workload_quadratic_upper_bound')
        # Effective base cost is proportional to original client data mass.
        np.testing.assert_allclose(z1['effective_fixed_cost'],z1['d']*z1['base_cost_total'])
        # Full enhancement cost above base is exactly proportional to the
        # amount of synthetic data required by addition-only balancing.
        enh=z1['d']*(z1['alpha']*z1['lam']+z1['beta']*z1['lam']**2)
        expected=z1['aigc_unit_cost']*z1['full_synthetic_n']/counts.sum()
        np.testing.assert_allclose(enh,expected,rtol=1e-12,atol=1e-12)

    def test_quadratic_calibration_upper_bounds_exact_additive_workload(self):
        counts=np.array([[900,100],[500,500]])
        p=counts/counts.sum(1)[:,None]
        d=counts.sum(1)/counts.sum()
        e=p-d@p
        lam=np.sqrt(2)*np.abs(e).sum(1)
        t=calibrated_types(counts,lam)
        for k in range(len(counts)):
            g=t['gamma'][k]
            A=t['alpha'][k]*lam[k]
            B=t['beta'][k]*lam[k]**2
            for u in np.linspace(0,1,51):
                exact=0. if g==0 else g*u/(1-g*u)
                quad=A*u+B*u*u
                self.assertGreaterEqual(quad+1e-12,exact)
            self.assertAlmostEqual(A+B,t['missing_ratio'][k],places=12)

    def test_additive_augmentation_retains_original(self):
        original=np.array([900,100]); pstar=np.array([.5,.5])
        expected={0.0:([900,100],[0,0]),.25:([900,225],[0,125]),
                  .75:([900,600],[0,500]),1.0:([900,900],[0,800])}
        for u,(final,added) in expected.items():
            target,got_final,got_added=additive_target_counts(original,pstar,u)
            np.testing.assert_array_equal(got_final,final)
            np.testing.assert_array_equal(got_added,added)
            self.assertTrue(np.all(got_final>=original))
            np.testing.assert_allclose(target,(1-u)*np.array([.9,.1])+u*pstar)
        target,final,added=additive_target_counts(original,pstar,.5)
        np.testing.assert_array_equal(final,[900,386])
        np.testing.assert_array_equal(added,[0,286])
        self.assertLess(np.max(np.abs(final/final.sum()-target)),2e-4)

    def test_local_epoch_step_accounting(self):
        self.assertEqual(local_epoch_steps(1000,128,2),16)
        self.assertEqual(local_epoch_steps(5000,128,2),80)
        self.assertEqual(local_epoch_steps(41000,32,2),2564)
        with self.assertRaises(ValueError): local_epoch_steps(0,32,2)

    def test_two_part_tariff_implements_target_and_pays_cost(self):
        z=self.interior(); active=np.ones(2,dtype=bool); u=np.array([.3,.8])
        fixed,rate=tariffs(z,active,u)
        got_active,got_u,utility=response(z,fixed,rate)
        np.testing.assert_array_equal(got_active,active)
        np.testing.assert_allclose(got_u,u,atol=1e-10)
        np.testing.assert_allclose(utility,0,atol=1e-10)
        q=u*z['lam']
        payment=z['d']*(fixed+rate*q)
        np.testing.assert_allclose(payment,implementation_payment(z,active,u),atol=1e-10)

    def test_continuous_weakly_dominates_three_state_in_J(self):
        for seed in range(10):
            rng=np.random.default_rng(seed)
            z=instance(rng.integers(1,100,(6,10)),seed=seed,budget_fraction=.4)
            cont=solve(z)
            disc=solve(z,discrete=True)
            self.assertEqual(cont['status'],'ok')
            self.assertEqual(disc['status'],'ok')
            self.assertLessEqual(cont['objective_normalized'],disc['objective_normalized']+1e-8)

    def test_discrete_full_allocation_has_same_payment_technology(self):
        z=self.interior(); active=np.ones(2,dtype=bool); u=np.ones(2)
        cont=implementation_payment(z,active,u)
        disc=implementation_payment(z,active,u)
        np.testing.assert_allclose(cont,disc,atol=0)
        f_c,r_c=tariffs(z,active,u,False)
        f_d,r_d=tariffs(z,active,u,True)
        self.assertTrue(np.all(r_c>=r_d))
        np.testing.assert_allclose(z['d']*(f_c+r_c*z['lam']),cont,atol=1e-10)
        np.testing.assert_allclose(z['d']*(f_d+r_d*z['lam']),disc,atol=1e-10)

    def test_independent_grid_allocation_cost(self):
        z=self.interior(); s=solve(z)
        grid=np.linspace(0,1,301); best=np.inf
        for a in grid:
            for b in grid:
                active=np.array([True,True]); u=np.array([a,b])
                if implementation_payment(z,active,u).sum()<=z['budget']+1e-12:
                    best=min(best,objective(z,active,u))
        self.assertLessEqual(s['objective_normalized'],best+1e-5)

    def test_zero_heterogeneity_and_infeasible(self):
        z=instance(np.array([[20,20],[30,30]]),budget_fraction=1)
        s=solve(z); self.assertEqual(s['status'],'ok')
        self.assertEqual(s['u'],[0.,0.])
        z['budget']=0; self.assertEqual(solve(z)['status'],'infeasible')

    def test_discrete_tie_break(self):
        for seed in range(20):
            rng=np.random.default_rng(seed)
            z=instance(rng.integers(1,100,(6,10)),seed=seed,budget_fraction=2)
            active=np.ones(6,dtype=bool); u=np.ones(6)
            fixed,rate=tariffs(z,active,u,discrete=True)
            got_active,got_u,_=response(z,fixed,rate,discrete=True)
            np.testing.assert_array_equal(got_active,active)
            np.testing.assert_array_equal(got_u,np.ones(6))
            result=solve(z,discrete=True)
            self.assertEqual(result['status'],'ok')

    def test_protocol_coefficients(self):
        np.testing.assert_allclose(coefficients(),[217.712669,26.444138,8.788034,.989603],rtol=1e-6)
        self.assertEqual(coefficients(steps=1)[1],0)
        self.assertEqual(coefficients(steps=1)[2],0)

if __name__=='__main__': unittest.main()
