import unittest
import numpy as np
from fusion_weights import fit_simplex, combine


class FusionWeightsTests(unittest.TestCase):
    def test_complementary_predictions(self):
        matrix=np.array([[0.,2.],[2.,0.],[1.,1.]])
        weights=fit_simplex(matrix,np.ones(3))
        np.testing.assert_allclose(weights,[.5,.5],atol=1e-7)
        np.testing.assert_allclose(combine(matrix,weights),np.ones(3),atol=1e-7)

    def test_boundary_and_determinism(self):
        matrix=np.array([[1.,5.],[2.,8.],[3.,9.]])
        a=fit_simplex(matrix,matrix[:,0])
        np.testing.assert_allclose(a,[1.,0.],atol=1e-7)
        np.testing.assert_array_equal(a,fit_simplex(matrix,matrix[:,0]))

    def test_reject_nonfinite_or_misaligned(self):
        for matrix,y in [(np.array([[np.nan,1.]]),np.ones(1)),(np.ones((2,2)),np.ones(3)),(np.ones((0,2)),np.ones(0))]:
            with self.assertRaises(ValueError):fit_simplex(matrix,y)
        with self.assertRaises(ValueError):combine(np.ones((2,2)),[1.,1.])


if __name__=='__main__':unittest.main()
