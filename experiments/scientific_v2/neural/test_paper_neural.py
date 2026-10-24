from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import unittest
import numpy as np
import pandas as pd
import torch
from paper_neural import PLEPreprocessor, PaperNeural, class_weights


class EncodingTests(unittest.TestCase):
    def test_train_only_knots_and_missing(self):
        fit = pd.DataFrame({'x':[0.,1.,2.,3.,np.nan], 'cat':['a','b','a',None,'a']})
        p = PLEPreprocessor().fit(fit)
        before = p.state()
        n,c = p.transform(pd.DataFrame({'x':[1e8,np.nan,np.inf], 'cat':['new',None,'a']}))
        self.assertEqual(before,p.state())
        self.assertTrue(np.isfinite(n).all() and np.isfinite(c).all())
        self.assertEqual(int(c[0].argmax()),2)
        self.assertEqual(int(c[1].argmax()),0)
        np.testing.assert_array_equal(n[1],n[2])

    def test_constant_and_all_missing(self):
        frame = pd.DataFrame({'constant':[1.,1.,1.], 'empty':[np.nan]*3})
        p = PLEPreprocessor().fit(frame)
        n,c = p.transform(frame)
        self.assertEqual(n.shape,(3,2,16))
        self.assertEqual(c.shape,(3,0))
        self.assertEqual(float(n.sum()),0.)

    def test_cap_rare_unseen(self):
        p = PLEPreprocessor(category_cap=5).fit(pd.DataFrame({'c':['a','a','b','c']}))
        _,x=p.transform(pd.DataFrame({'c':['a','b','c','d',None]}))
        np.testing.assert_array_equal(x.argmax(1),[3,4,1,2,0])

    def test_shapes_gradient_and_member_diversity(self):
        torch.manual_seed(42)
        n=torch.randn(5,2,16); c=torch.eye(5)
        for members in [1,8]:
            m=PaperNeural(2,[5],3,members)
            m.eval(); output=m(n,c)
            self.assertEqual(tuple(output.shape),(5,members,3))
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            if members>1:self.assertFalse(torch.allclose(output[:,0],output[:,1]))

    def test_normalized_weights(self):
        y=np.array([0,0,0,1,2])
        w=class_weights(y,3).numpy()
        self.assertAlmostEqual(float(np.mean(w[y])),1.,places=6)
        self.assertGreater(w[1],w[0])

if __name__=='__main__':unittest.main()
