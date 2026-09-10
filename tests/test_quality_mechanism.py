import unittest
import numpy as np

from baqp.mechanism_quality import instance, solve, objective, objective_components


class GeneratorAwareMechanismTests(unittest.TestCase):
    def test_full_is_not_perfect_when_generator_mismatch_positive(self):
        counts=np.array([[90,10],[10,90]])
        z=instance(counts,budget_fraction=2.0,generator_gap_ratio=.3)
        active=np.ones(2,dtype=bool)
        full=objective_components(z,active,np.ones(2))
        self.assertGreater(full['generator'],0.0)
        self.assertGreater(full['total'],full['noise'])

    def test_zero_gap_removes_generator_penalty(self):
        counts=np.array([[90,10],[10,90]])
        z=instance(counts,budget_fraction=2.0,generator_gap_ratio=0.0)
        active=np.ones(2,dtype=bool)
        full=objective_components(z,active,np.ones(2))
        self.assertAlmostEqual(full['generator'],0.0,places=14)
        self.assertAlmostEqual(full['total'],full['noise'],places=14)

    def test_positive_gap_can_create_strict_interior_optimum(self):
        counts=np.array([[90,10],[10,90]])
        z=instance(counts,budget_fraction=2.0,generator_gap_ratio=.3)
        result=solve(z)
        self.assertEqual(result['status'],'ok')
        active=np.asarray(result['active'],dtype=bool)
        u=np.asarray(result['u'],dtype=float)
        self.assertTrue(np.any(active & (u>1e-4) & (u<1-1e-4)))
        # The continuous allocation should beat both all-raw and all-full on
        # the same active participant set when mismatch is material.
        raw=np.where(active,0.0,0.0)
        full=np.where(active,1.0,0.0)
        self.assertLess(result['objective_normalized'],objective(z,active,raw)-1e-8)
        self.assertLess(result['objective_normalized'],objective(z,active,full)-1e-8)

    def test_continuous_weakly_dominates_three_state(self):
        rng=np.random.default_rng(7)
        for _ in range(5):
            counts=rng.integers(1,100,(6,10))
            z=instance(counts,budget_fraction=.4,generator_gap_ratio=.3)
            cont=solve(z)
            disc=solve(z,discrete=True)
            self.assertEqual(cont['status'],'ok')
            self.assertEqual(disc['status'],'ok')
            self.assertLessEqual(cont['objective_normalized'],disc['objective_normalized']+1e-8)


if __name__=='__main__':
    unittest.main()
