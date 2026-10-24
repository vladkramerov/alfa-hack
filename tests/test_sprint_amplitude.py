from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import sprint_amplitude as amplitude


class AmplitudeTests(unittest.TestCase):
    def test_exact_no_intercept_optimum_and_frozen_application(self):
        base=np.array([10.,20.,30.]);pool=np.array([12.,24.,36.]);y=np.array([13.,26.,39.])
        fitted=amplitude.fit_amplitude(pool,base,y)
        self.assertEqual(fitted,1.5)
        np.testing.assert_array_equal(amplitude.apply_amplitude(pool,base,fitted),y)
        self.assertEqual(amplitude.fit_amplitude([1.,1.],[0.,0.],[3.,1.]),2.)

    def test_boundaries_zero_delta_and_nonnegative_projection(self):
        self.assertEqual(amplitude.fit_amplitude([2.,3.],[1.,1.],[0.,0.]),0.)
        self.assertEqual(amplitude.fit_amplitude([2.,3.],[1.,1.],[100.,100.]),2.)
        self.assertEqual(amplitude.fit_amplitude([1.,2.],[1.,2.],[100.,0.]),1.)
        np.testing.assert_array_equal(amplitude.apply_amplitude([0.,1.],[10.,10.],2.),[0.,0.])
        for bad in (np.nan,-.1,2.1):
            with self.assertRaises(ValueError): amplitude.apply_amplitude([1.],[0.],bad)
        with self.assertRaises(ValueError): amplitude.fit_amplitude([np.nan],[0.],[1.])
        with self.assertRaises(ValueError): amplitude.fit_amplitude([1.,2.],[0.],[1.,2.])

    def test_claim_is_saved_before_inner_fit_and_outer_never_calls_optimizer(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as directory:
            path=Path(directory);ledger=path/'ledger.md';ledger.write_text('HYP-102 registered',encoding='utf8')
            history=amplitude.Memory(path/'history.json',ledger)
            real_fit=amplitude.fit_amplitude
            def observe(*args):
                records=history.records()
                self.assertEqual(len(records),1);self.assertEqual(records[0]['status'],'started')
                return real_fit(*args)
            with patch.object(amplitude,'fit_amplitude',side_effect=observe):
                value,claim=amplitude.select_amplitude('inner',[12.,24.],[10.,20.],[13.,26.],
                    configuration={'hypothesis':'HYP-102'},history=history)
            self.assertEqual(value,1.5);self.assertIsNotNone(claim)
            state={'status':'completed','eligible_for_outer':True,'amplitude':value}
            with patch.object(amplitude,'fit_amplitude',side_effect=AssertionError('outer optimizer forbidden')):
                outer,claim=amplitude.select_amplitude('outer',[99.],[20.],[-999.],frozen=state)
            self.assertEqual(outer,1.5);self.assertIsNone(claim)
            with self.assertRaises(ValueError):
                amplitude.select_amplitude('outer',[1.],[0.],[1.],frozen={**state,'eligible_for_outer':False})

    def test_cache_alignment_requires_exact_positions_y_and_split(self):
        pool={'positions':np.array([1,2]),'y':np.array([10.,20.]),'prediction':np.array([11.,19.])}
        transport={'positions':np.array([1,2]),'y':np.array([10.,20.]),'reference':np.array([8.,22.])}
        amplitude.validate_cached(pool,transport,{'fit':np.array([0]),'valid':np.array([1,2])})
        with self.assertRaises(ValueError):
            amplitude.validate_cached(pool,{**transport,'positions':np.array([2,1])},
                                      {'fit':np.array([0]),'valid':np.array([1,2])})


if __name__=='__main__':unittest.main()
