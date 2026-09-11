import unittest
import numpy as np
from baqp.data_absolute import (split_train_fixed_size,absolute_target_counts,
                                optimal_augmented_distribution,repair_quantity)
from baqp.mechanism_absolute import instance,budget_implied_upper,client_cost


class AbsoluteQuantityTests(unittest.TestCase):
    def test_fixed_size_partition_preserves_margins(self):
        labels=np.repeat(np.arange(3),120)
        parts,val,counts=split_train_fixed_size(labels,3,clients=6,alpha=.3,seed=7,val_fraction=.1)
        self.assertLessEqual(max(map(len,parts))-min(map(len,parts)),1)
        self.assertEqual(len(np.unique(np.concatenate(parts))),sum(map(len,parts)))
        self.assertTrue(np.array_equal(counts.sum(0),np.array([108,108,108])))
        self.assertEqual(len(val),36)

    def test_absolute_plan_has_no_ratio_cap(self):
        c=np.array([90,10]); p=np.array([.5,.5])
        target,final,add=absolute_target_counts(c,p,250)
        self.assertEqual(int(add.sum()),250)
        self.assertTrue(np.all(final>=c))
        self.assertTrue(np.isclose(target.sum(),1))
        self.assertGreater(250,c.sum())

    def test_repair_is_reference_not_domain_limit(self):
        c=np.array([90.,10.]); p=np.array([.5,.5])
        qr=repair_quantity(c,p)
        self.assertEqual(qr,80.)
        d1,_=optimal_augmented_distribution(c,p,qr)
        d2,_=optimal_augmented_distribution(c,p,qr+200)
        self.assertTrue(np.allclose(d1,p))
        self.assertTrue(np.allclose(d2,p))

    def test_solver_upper_bound_is_budget_implied_and_can_exceed_real_n(self):
        counts=np.array([[90,10],[10,90]],dtype=float)
        z=instance(counts,budget_fraction=3.0,aigc_convex_cost=.1)
        active=np.ones(2,dtype=bool)
        high=budget_implied_upper(z,active)
        self.assertTrue(np.all(high>z['n']))
        rem=z['budget']-z['base_cost'].sum()
        k=0
        inc=client_cost(z,np.array([high[k],0.]),active).sum()-z['base_cost'].sum()
        self.assertAlmostEqual(float(inc),float(rem),places=8)


if __name__=='__main__': unittest.main()
