import unittest
import numpy as np
from experiments.sprint_blend import convex_weight

class BlendTests(unittest.TestCase):
    def test_analytical_weight_recovers_interior_optimum(self):
        a=np.array([0.,1.,2.]);b=np.array([2.,0.,0.]);y=.3*a+.7*b
        self.assertAlmostEqual(convex_weight(a,b,y),.3)
    def test_equal_predictions_and_boundary(self):
        self.assertEqual(convex_weight([1,2],[1,2],[1,2]),1.)
        self.assertEqual(convex_weight([1,2],[3,4],[0,1]),1.)
