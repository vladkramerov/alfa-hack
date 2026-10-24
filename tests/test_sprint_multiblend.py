import unittest
import numpy as np
from experiments.sprint_multiblend import fit_weights

class MultiBlendTests(unittest.TestCase):
    def test_recovers_complementary_weights_and_simplex(self):
        p=np.eye(3);w=fit_weights(p,p@np.array([.2,.3,.5]))
        np.testing.assert_allclose(w,[.2,.3,.5],atol=1e-5)
        self.assertAlmostEqual(sum(w),1);self.assertTrue(np.all(w>=0))
    def test_boundary_better_than_bad_components(self):
        p=np.column_stack([np.array([1.,2,3]),np.array([5.,6,7]),np.array([4.,5,6])])
        np.testing.assert_allclose(fit_weights(p,[1,2,3]),[1,0,0],atol=1e-5)
