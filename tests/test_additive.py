import unittest
import numpy as np
import torch
from baqp.data import additive_target_counts
from baqp.experiment import build_augmented_indices, draw_additive_batch


class AdditivePipelineTests(unittest.TestCase):
    def test_retained_ids_and_uniform_multiset_sampling(self):
        y=torch.tensor([0]*900+[1]*100)
        x=torch.zeros(1000,1,1,1,dtype=torch.uint8)
        sy=torch.tensor([0]*1000+[1]*1000)
        sx=torch.ones(2000,1,1,1,dtype=torch.uint8)
        ids=np.arange(1000)
        plans=build_augmented_indices([ids],np.array([[900,100]]),np.array([.5,.5]),
                                      dict(active=[True],u=[1.]),sy,2026)
        plan=plans[0]
        np.testing.assert_array_equal(plan['real_ids'],ids)
        self.assertEqual(plan['synthetic_n'],800)
        self.assertTrue(torch.all(sy[plan['synthetic_ids']]==1))
        bx,by,draws=draw_additive_batch(plan,x,y,sx,sy,20000,np.random.default_rng(4))
        self.assertEqual(int(bx.sum()),draws)
        # u=1 balances labels while retaining a 1000/1800 real sampling share.
        self.assertAlmostEqual(draws/20000,800/1800,delta=.015)
        self.assertAlmostEqual(float(by.float().mean()),.5,delta=.015)

    def test_zero_additions_and_cache_reuse(self):
        ids=np.arange(10); counts=np.array([[9,1]])
        sy=torch.tensor([0,1])
        for u,expected in [(0.,0),(1.,8)]:
            plan=build_augmented_indices([ids],counts,np.array([.5,.5]),
                                         dict(active=[True],u=[u]),sy,0)[0]
            np.testing.assert_array_equal(plan['real_ids'],ids)
            self.assertEqual(plan['synthetic_n'],expected)
            if expected:
                np.testing.assert_array_equal(plan['synthetic_ids'],np.ones(8,dtype=int))

    def test_zero_support_and_rounding(self):
        target,final,added=additive_target_counts([9,1,0],[.5,.5,0],.5)
        self.assertEqual(final[2],0)
        self.assertTrue(np.all(final>=np.array([9,1,0])))
        self.assertLessEqual(np.max(np.abs(final/final.sum()-target)),1/final.sum())
        with self.assertRaises(ValueError):
            additive_target_counts([9,1],[0,1],1.)


if __name__=='__main__': unittest.main()
