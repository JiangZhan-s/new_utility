import unittest
import numpy as np

from baqp.mechanism_joint_absolute import instance, objective_components


class JointAbsoluteMechanismTests(unittest.TestCase):
    def setUp(self):
        # Equal-size, heterogeneous clients.
        self.counts=np.array([
            [90,5,5],
            [5,90,5],
            [5,5,90],
            [40,30,30],
        ],dtype=float)
        self.z=instance(self.counts,budget_fraction=1.0,generator_gap_ratio=0.05)

    def test_single_repaired_client_does_not_zero_global_bias(self):
        active=np.array([True,False,False,False])
        q=np.zeros(4); q[0]=self.z['repair_q'][0]
        pieces=objective_components(self.z,active,q)
        self.assertGreater(pieces['missing'],0.0)
        self.assertAlmostEqual(pieces['active_mass'],0.25,places=12)
        self.assertGreater(pieces['total'],0.0)

    def test_all_raw_clients_remove_missing_mass_and_aggregate_label_bias(self):
        active=np.ones(4,dtype=bool); q=np.zeros(4)
        pieces=objective_components(self.z,active,q)
        self.assertAlmostEqual(pieces['missing'],0.0,places=12)
        self.assertAlmostEqual(pieces['label'],0.0,places=12)
        self.assertGreater(pieces['local'],0.0)

    def test_participation_is_a_decision_variable(self):
        a1=np.array([True,False,False,False])
        a2=np.array([True,True,False,False])
        p1=objective_components(self.z,a1,np.zeros(4))
        p2=objective_components(self.z,a2,np.zeros(4))
        self.assertGreater(p1['missing'],p2['missing'])


if __name__=='__main__':
    unittest.main()
